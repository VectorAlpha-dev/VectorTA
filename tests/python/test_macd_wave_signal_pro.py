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
    open_ = 100.0 + idx * 0.08 + np.sin(idx * 0.05) * 0.7
    close = open_ + np.cos(idx * 0.09) * 0.9
    high = np.maximum(open_, close) + 0.55 + np.abs(np.sin(idx * 0.03)) * 0.2
    low = np.minimum(open_, close) - 0.55 - np.abs(np.cos(idx * 0.05)) * 0.2
    return open_, high, low, close


class TestMacdWaveSignalPro:
    def test_output_contract(self):
        open_, high, low, close = sample_ohlc()
        diff, dea, macd_histogram, line_convergence, buy_signal, sell_signal = (
            mp.macd_wave_signal_pro(open_, high, low, close)
        )

        assert len(diff) == len(close)
        assert len(dea) == len(close)
        assert len(macd_histogram) == len(close)
        assert len(line_convergence) == len(close)
        assert len(buy_signal) == len(close)
        assert len(sell_signal) == len(close)
        assert np.isfinite(diff).any()
        assert np.isfinite(line_convergence).any()
        valid_buy = buy_signal[np.isfinite(buy_signal)]
        valid_sell = sell_signal[np.isfinite(sell_signal)]
        assert np.all((valid_buy == 0.0) | (valid_buy == 1.0))
        assert np.all((valid_sell == 0.0) | (valid_sell == 1.0))

    def test_kernel_parity(self):
        open_, high, low, close = sample_ohlc(192)
        auto = mp.macd_wave_signal_pro(open_, high, low, close)
        scalar = mp.macd_wave_signal_pro(open_, high, low, close, kernel="scalar")
        for a, b in zip(auto, scalar):
            np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12, equal_nan=True)

    def test_invalid_inputs(self):
        open_, high, low, close = sample_ohlc(32)

        with pytest.raises(ValueError, match="Not enough valid data"):
            mp.macd_wave_signal_pro(open_, high, low, close)

        with pytest.raises(ValueError, match="Inconsistent slice lengths"):
            mp.macd_wave_signal_pro(open_[:-1], high, low, close)

        all_nan = np.full(64, np.nan, dtype=np.float64)
        with pytest.raises(ValueError, match="All values are NaN"):
            mp.macd_wave_signal_pro(all_nan, all_nan, all_nan, all_nan)

    def test_stream_matches_batch_with_reset(self):
        open_, high, low, close = sample_ohlc(220)
        open_[90] = np.nan
        high[90] = np.nan
        low[90] = np.nan
        close[90] = np.nan

        batch = mp.macd_wave_signal_pro(open_, high, low, close)
        stream = mp.MacdWaveSignalProStream()

        streamed = [[] for _ in range(6)]
        for values in zip(open_, high, low, close):
            out = stream.update(*map(float, values))
            if out is None:
                for series in streamed:
                    series.append(np.nan)
            else:
                for series, value in zip(streamed, out):
                    series.append(value)

        assert stream.warmup_period == 39
        for batch_series, stream_series in zip(batch, streamed):
            np.testing.assert_allclose(
                batch_series, stream_series, rtol=1e-12, atol=1e-12, equal_nan=True
            )

    def test_batch_single_row_matches_single(self):
        open_, high, low, close = sample_ohlc(128)
        batch = mp.macd_wave_signal_pro_batch(open_, high, low, close)
        single = mp.macd_wave_signal_pro(open_, high, low, close)

        assert batch["rows"] == 1
        assert batch["cols"] == len(close)
        assert batch["diff"].shape == (1, len(close))
        assert batch["dea"].shape == (1, len(close))
        assert batch["macd_histogram"].shape == (1, len(close))
        assert batch["line_convergence"].shape == (1, len(close))
        assert batch["buy_signal"].shape == (1, len(close))
        assert batch["sell_signal"].shape == (1, len(close))

        keys = [
            "diff",
            "dea",
            "macd_histogram",
            "line_convergence",
            "buy_signal",
            "sell_signal",
        ]
        for key, expected in zip(keys, single):
            np.testing.assert_allclose(
                batch[key][0], expected, rtol=1e-12, atol=1e-12, equal_nan=True
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
