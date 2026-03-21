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


def output_keys():
    return [
        "basis",
        "trend",
        "upper_0618",
        "upper_1000",
        "upper_1618",
        "upper_2618",
        "lower_0618",
        "lower_1000",
        "lower_1618",
        "lower_2618",
        "tp_long_band",
        "tp_short_band",
        "long_entry",
        "short_entry",
        "rejection_long",
        "rejection_short",
        "long_bounce",
        "short_bounce",
    ]


class TestFibonacciEntryBands:
    def test_output_contract(self):
        open_, high, low, close = sample_ohlc()
        outputs = mp.fibonacci_entry_bands(
            open_, high, low, close, source="hlc3", length=21, atr_length=14, use_atr=True
        )

        assert set(outputs.keys()) == set(output_keys())
        for key in output_keys():
            assert len(outputs[key]) == len(close)

        assert np.isfinite(outputs["basis"]).any()
        assert np.isfinite(outputs["tp_long_band"]).any()
        assert np.isfinite(outputs["tp_short_band"]).any()

        trend = outputs["trend"]
        trend_valid = trend[np.isfinite(trend)]
        assert np.all((trend_valid == -1.0) | (trend_valid == 0.0) | (trend_valid == 1.0))

        for key in [
            "long_entry",
            "short_entry",
            "rejection_long",
            "rejection_short",
            "long_bounce",
            "short_bounce",
        ]:
            signal = outputs[key]
            valid = signal[np.isfinite(signal)]
            assert np.all((valid == 0.0) | (valid == 1.0))

    def test_invalid_parameters(self):
        open_, high, low, close = sample_ohlc(32)

        with pytest.raises(ValueError, match="Invalid source"):
            mp.fibonacci_entry_bands(open_, high, low, close, source="bad")

        with pytest.raises(ValueError, match="Invalid length"):
            mp.fibonacci_entry_bands(open_, high, low, close, length=0)

        with pytest.raises(ValueError, match="Invalid TP aggressiveness"):
            mp.fibonacci_entry_bands(open_, high, low, close, tp_aggressiveness="bad")

    def test_stream_matches_batch_with_reset(self):
        open_, high, low, close = sample_ohlc(220)
        open_[90] = np.nan
        high[90] = np.nan
        low[90] = np.nan
        close[90] = np.nan

        batch = mp.fibonacci_entry_bands(
            open_,
            high,
            low,
            close,
            source="hlc3",
            length=16,
            atr_length=11,
            use_atr=True,
            tp_aggressiveness="high",
        )
        stream = mp.FibonacciEntryBandsStream(
            source="hlc3",
            length=16,
            atr_length=11,
            use_atr=True,
            tp_aggressiveness="high",
        )

        streamed = {key: [] for key in output_keys()}
        for values in zip(open_, high, low, close):
            out = stream.update(*map(float, values))
            if out is None:
                for key in output_keys():
                    streamed[key].append(np.nan)
            else:
                for key in output_keys():
                    streamed[key].append(out[key])

        assert stream.warmup_period == 10
        for key in output_keys():
            np.testing.assert_allclose(
                batch[key],
                streamed[key],
                rtol=1e-12,
                atol=1e-12,
                equal_nan=True,
            )

    def test_batch_single_row_matches_single(self):
        open_, high, low, close = sample_ohlc(128)
        batch = mp.fibonacci_entry_bands_batch(
            open_,
            high,
            low,
            close,
            length_range=(17, 17, 0),
            atr_length_range=(10, 10, 0),
            source="hlc3",
            use_atr=True,
            tp_aggressiveness="medium",
        )
        single = mp.fibonacci_entry_bands(
            open_,
            high,
            low,
            close,
            source="hlc3",
            length=17,
            atr_length=10,
            use_atr=True,
            tp_aggressiveness="medium",
        )

        assert batch["rows"] == 1
        assert batch["cols"] == len(close)
        np.testing.assert_array_equal(batch["lengths"], np.array([17], dtype=np.uint64))
        np.testing.assert_array_equal(batch["atr_lengths"], np.array([10], dtype=np.uint64))
        np.testing.assert_array_equal(batch["use_atr_flags"], np.array([True]))
        assert batch["sources"] == ["hlc3"]
        assert batch["tp_aggressiveness_values"] == ["medium"]

        for key in output_keys():
            np.testing.assert_allclose(
                batch[key][0], single[key], rtol=1e-12, atol=1e-12, equal_nan=True
            )

    def test_batch_metadata(self):
        open_, high, low, close = sample_ohlc(120)
        batch = mp.fibonacci_entry_bands_batch(
            open_,
            high,
            low,
            close,
            length_range=(16, 18, 2),
            atr_length_range=(10, 12, 2),
            source="hl2",
            use_atr=False,
            tp_aggressiveness="low",
        )

        assert batch["rows"] == 4
        assert batch["cols"] == len(close)
        for key in output_keys():
            assert batch[key].shape == (4, len(close))

        np.testing.assert_array_equal(
            batch["lengths"], np.array([16, 16, 18, 18], dtype=np.uint64)
        )
        np.testing.assert_array_equal(
            batch["atr_lengths"], np.array([10, 12, 10, 12], dtype=np.uint64)
        )
        np.testing.assert_array_equal(batch["use_atr_flags"], np.array([False, False, False, False]))
        assert batch["sources"] == ["hl2", "hl2", "hl2", "hl2"]
        assert batch["tp_aggressiveness_values"] == ["low", "low", "low", "low"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
