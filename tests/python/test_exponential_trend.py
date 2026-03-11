import numpy as np
import pytest

try:
    import vector_ta as ta_indicators
except ImportError:
    try:
        import my_project as ta_indicators
    except ImportError:
        pytest.skip(
            "Python module not built. Run 'maturin develop --features python' first",
            allow_module_level=True,
        )


def sample_hlc():
    high = []
    low = []
    close = []

    for i in range(60):
        close.append(100.0 + i * 0.65 + ((i % 5) - 2) * 0.08)
    for i in range(60, 100):
        x = i - 60
        close.append(139.0 + x * 0.05 - (i % 4) * 0.12)
    for i in range(100, 120):
        x = i - 100
        close.append(140.0 + x * 0.55)
    for i in range(120, 140):
        x = i - 120
        close.append(150.0 - x * 2.4)
    for i in range(140, 160):
        x = i - 140
        close.append(102.0 + x * 1.8)

    for i, c in enumerate(close):
        wiggle = (i % 3) * 0.15
        high.append(c + 1.6 + wiggle)
        low.append(c - 1.5 - wiggle * 0.8)

    return tuple(np.asarray(values, dtype=float) for values in (high, low, close))


def test_exponential_trend_outputs_present():
    high, low, close = sample_hlc()
    result = ta_indicators.exponential_trend(high, low, close)
    assert set(result.keys()) == {
        "uptrend_base",
        "downtrend_base",
        "uptrend_extension",
        "downtrend_extension",
        "bullish_change",
        "bearish_change",
    }
    assert np.isfinite(result["uptrend_base"]).any()
    assert np.isfinite(result["downtrend_base"]).any()


def test_exponential_trend_stream_matches_batch():
    high, low, close = sample_hlc()
    batch = ta_indicators.exponential_trend(
        high,
        low,
        close,
        exp_rate=0.00003,
        initial_distance=4.0,
        width_multiplier=1.0,
    )
    stream = ta_indicators.ExponentialTrendStream(
        exp_rate=0.00003,
        initial_distance=4.0,
        width_multiplier=1.0,
    )

    values = [np.full(close.size, np.nan) for _ in range(6)]
    for idx, item in enumerate(zip(high, low, close)):
        result = stream.update(*map(float, item))
        if result is not None:
            for series, value in zip(values, result):
                series[idx] = value

    for actual, key in zip(
        values,
        [
            "uptrend_base",
            "downtrend_base",
            "uptrend_extension",
            "downtrend_extension",
            "bullish_change",
            "bearish_change",
        ],
    ):
        np.testing.assert_allclose(actual, batch[key], rtol=0.0, atol=0.0, equal_nan=True)


def test_exponential_trend_batch_matches_single():
    high, low, close = sample_hlc()
    batch = ta_indicators.exponential_trend_batch(
        high,
        low,
        close,
        exp_rate_range=(0.00003, 0.00003, 0.0),
        initial_distance_range=(4.0, 4.1, 0.1),
        width_multiplier_range=(1.0, 1.0, 0.0),
    )
    single = ta_indicators.exponential_trend(
        high, low, close, exp_rate=0.00003, initial_distance=4.0, width_multiplier=1.0
    )
    assert batch["rows"] == 2
    assert batch["cols"] == close.size
    np.testing.assert_allclose(
        batch["uptrend_base"][0], single["uptrend_base"], rtol=0.0, atol=0.0, equal_nan=True
    )


def test_exponential_trend_invalid_exp_rate():
    high, low, close = sample_hlc()
    with pytest.raises(ValueError, match="exp_rate"):
        ta_indicators.exponential_trend(
            high, low, close, exp_rate=0.8, initial_distance=4.0, width_multiplier=1.0
        )
