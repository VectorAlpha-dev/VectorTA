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
    for i in range(180):
        if i < 60:
            base = 100.0 + i * 0.35
        elif i < 120:
            base = 121.0 - (i - 60) * 0.22
        else:
            base = 108.0 + (i - 120) * 0.42
        wiggle = ((i % 7) - 3) * 0.18
        c = base + wiggle
        o = c - ((i % 5) - 2) * 0.11
        open_.append(o)
        close.append(c)
        high.append(max(c, o) + 1.2 + (i % 3) * 0.07)
        low.append(min(c, o) - 1.1 - (i % 4) * 0.05)
        volume.append(1000.0 + (i % 9) * 37.0 + i * 4.0)
    return tuple(np.asarray(values, dtype=float) for values in (open_, high, low, close, volume))


def test_trend_flow_trail_outputs_present():
    open_, high, low, close, volume = sample_ohlcv()
    result = ta_indicators.trend_flow_trail(open_, high, low, close, volume)
    assert set(result.keys()) == {
        "alpha_trail",
        "alpha_trail_bullish",
        "alpha_trail_bearish",
        "alpha_dir",
        "mfi",
        "tp_upper",
        "tp_lower",
        "alpha_trail_bullish_switch",
        "alpha_trail_bearish_switch",
        "mfi_overbought",
        "mfi_oversold",
        "mfi_cross_up_mid",
        "mfi_cross_down_mid",
        "price_cross_alpha_trail_up",
        "price_cross_alpha_trail_down",
        "mfi_above_90",
        "mfi_below_10",
    }
    assert np.isfinite(result["alpha_trail"]).any()
    assert np.isfinite(result["mfi"]).any()


def test_trend_flow_trail_stream_matches_batch():
    open_, high, low, close, volume = sample_ohlcv()
    batch = ta_indicators.trend_flow_trail(open_, high, low, close, volume)
    stream = ta_indicators.TrendFlowTrailStream(alpha_length=33, alpha_multiplier=3.3, mfi_length=14)
    alpha_trail = np.full(close.size, np.nan)
    for i, values in enumerate(zip(open_, high, low, close, volume)):
        result = stream.update(*map(float, values))
        if result is not None:
            alpha_trail[i] = result[0]
    np.testing.assert_allclose(alpha_trail, batch["alpha_trail"], rtol=0.0, atol=0.0, equal_nan=True)


def test_trend_flow_trail_batch_matches_single():
    open_, high, low, close, volume = sample_ohlcv()
    batch = ta_indicators.trend_flow_trail_batch(
        open_,
        high,
        low,
        close,
        volume,
        alpha_length_range=(33, 33, 0),
        alpha_multiplier_range=(3.3, 3.4, 0.1),
        mfi_length_range=(14, 14, 0),
    )
    single = ta_indicators.trend_flow_trail(open_, high, low, close, volume)
    assert batch["rows"] == 2
    assert batch["cols"] == close.size
    np.testing.assert_allclose(
        batch["alpha_trail"][0], single["alpha_trail"], rtol=0.0, atol=0.0, equal_nan=True
    )


def test_trend_flow_trail_invalid_alpha_multiplier():
    open_, high, low, close, volume = sample_ohlcv()
    with pytest.raises(ValueError, match="alpha_multiplier"):
        ta_indicators.trend_flow_trail(
            open_, high, low, close, volume, alpha_length=33, alpha_multiplier=0.0, mfi_length=14
        )
