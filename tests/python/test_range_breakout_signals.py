import numpy as np
import pytest

try:
    import vector_ta as ta_indicators
except ImportError:
    try:
        import vector_ta as ta_indicators
    except ImportError:
        pytest.skip(
            "Python module not built. Run 'maturin develop --features python' first",
            allow_module_level=True,
        )


def sample_ohlcv():
    open_, high, low, close, volume = [], [], [], [], []

    for i in range(24):
        base = 100.0 + i * 0.35
        o = base - 1.4
        c = base + (1.7 if i % 2 == 0 else -1.6)
        open_.append(o)
        close.append(c)
        high.append(max(o, c) + 1.1)
        low.append(min(o, c) - 1.0)
        volume.append(900.0 + i * 8.0)

    for i in range(24, 36):
        base = 108.5 + (i - 24) * 0.03
        o = base - 0.03
        c = base + 0.03
        open_.append(o)
        close.append(c)
        high.append(base + 0.18)
        low.append(base - 0.18)
        volume.append(1600.0 + i * 6.0)

    open_.append(109.2)
    high.append(112.4)
    low.append(109.0)
    close.append(112.1)
    volume.append(2200.0)

    for i in range(37, 60):
        base = 111.8 - (i - 37) * 0.08
        o = base + 0.8
        c = base - 0.9
        open_.append(o)
        close.append(c)
        high.append(max(o, c) + 0.9)
        low.append(min(o, c) - 0.8)
        volume.append(1100.0 + i * 5.0)

    for i in range(60, 72):
        base = 104.7 - (i - 60) * 0.02
        o = base + 0.02
        c = base - 0.02
        open_.append(o)
        close.append(c)
        high.append(base + 0.16)
        low.append(base - 0.16)
        volume.append(1750.0 + i * 5.0)

    open_.append(103.9)
    high.append(104.0)
    low.append(100.6)
    close.append(100.9)
    volume.append(2400.0)

    for i in range(73, 96):
        base = 101.4 + (i - 73) * 0.06
        o = base - 0.3
        c = base + 0.25
        open_.append(o)
        close.append(c)
        high.append(max(o, c) + 0.45)
        low.append(min(o, c) - 0.45)
        volume.append(1200.0 + i * 3.0)

    return tuple(np.asarray(values, dtype=float) for values in (open_, high, low, close, volume))


def test_range_breakout_signals_outputs_present():
    open_, high, low, close, volume = sample_ohlcv()
    result = ta_indicators.range_breakout_signals(open_, high, low, close, volume)
    assert set(result.keys()) == {
        "range_top",
        "range_bottom",
        "bullish",
        "extra_bullish",
        "bearish",
        "extra_bearish",
    }
    assert np.isfinite(result["range_top"]).any()
    assert np.isfinite(result["range_bottom"]).any()


def test_range_breakout_signals_stream_matches_batch():
    open_, high, low, close, volume = sample_ohlcv()
    batch = ta_indicators.range_breakout_signals(
        open_, high, low, close, volume, range_length=18, confirmation_length=4
    )
    stream = ta_indicators.RangeBreakoutSignalsStream(
        range_length=18, confirmation_length=4
    )

    values = [np.full(close.size, np.nan) for _ in range(6)]
    for idx, item in enumerate(zip(open_, high, low, close, volume)):
        result = stream.update(*map(float, item))
        if result is not None:
            for series, value in zip(values, result):
                series[idx] = value

    for actual, key in zip(
        values,
        [
            "range_top",
            "range_bottom",
            "bullish",
            "extra_bullish",
            "bearish",
            "extra_bearish",
        ],
    ):
        np.testing.assert_allclose(actual, batch[key], rtol=0.0, atol=0.0, equal_nan=True)


def test_range_breakout_signals_batch_matches_single():
    open_, high, low, close, volume = sample_ohlcv()
    batch = ta_indicators.range_breakout_signals_batch(
        open_,
        high,
        low,
        close,
        volume,
        range_length_range=(20, 21, 1),
        confirmation_length_range=(5, 5, 0),
    )
    single = ta_indicators.range_breakout_signals(
        open_, high, low, close, volume, range_length=20, confirmation_length=5
    )
    assert batch["rows"] == 2
    assert batch["cols"] == close.size
    np.testing.assert_allclose(
        batch["range_top"][0], single["range_top"], rtol=0.0, atol=0.0, equal_nan=True
    )


def test_range_breakout_signals_invalid_confirmation_length():
    open_, high, low, close, volume = sample_ohlcv()
    with pytest.raises(ValueError, match="confirmation_length"):
        ta_indicators.range_breakout_signals(
            open_, high, low, close, volume, range_length=20, confirmation_length=0
        )
