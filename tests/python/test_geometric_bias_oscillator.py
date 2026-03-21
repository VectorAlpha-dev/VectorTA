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

from test_utils import assert_close


def trend_data(size, slope):
    high = np.empty(size, dtype=np.float64)
    low = np.empty(size, dtype=np.float64)
    close = np.empty(size, dtype=np.float64)
    for i in range(size):
        c = 100.0 + slope * i
        close[i] = c
        high[i] = c + 1.0
        low[i] = c - 1.0
    return high, low, close


def mixed_data(size):
    high = np.empty(size, dtype=np.float64)
    low = np.empty(size, dtype=np.float64)
    close = np.empty(size, dtype=np.float64)
    for i in range(size):
        x = float(i)
        c = 100.0 + 0.35 * x + np.sin(x * 0.27) * 4.0 + np.cos(x * 0.11) * 1.5
        close[i] = c
        high[i] = c + 1.2 + (i % 3) * 0.1
        low[i] = c - 1.1 - (i % 2) * 0.1
    return high, low, close


def test_geometric_bias_oscillator_increasing_trend_reaches_hundred():
    high, low, close = trend_data(160, 1.0)
    result = ta_indicators.geometric_bias_oscillator(
        high, low, close, length=20, multiplier=2.0, atr_length=14, smooth=3
    )
    start = max(20, 14) + 3 - 2
    assert np.allclose(result[start:], 100.0, atol=1e-12)


def test_geometric_bias_oscillator_decreasing_trend_reaches_negative_hundred():
    high, low, close = trend_data(160, -1.0)
    result = ta_indicators.geometric_bias_oscillator(
        high, low, close, length=20, multiplier=2.0, atr_length=14, smooth=3
    )
    start = max(20, 14) + 3 - 2
    assert np.allclose(result[start:], -100.0, atol=1e-12)


def test_geometric_bias_oscillator_nan_gap_restart():
    high, low, close = mixed_data(180)
    high[120] = np.nan
    low[120] = np.nan
    close[120] = np.nan
    result = ta_indicators.geometric_bias_oscillator(
        high, low, close, length=18, multiplier=2.0, atr_length=10, smooth=3
    )
    restart_end = min(result.size, 120 + max(18, 10) + 3 - 1)
    assert np.all(np.isnan(result[120:restart_end]))


def test_geometric_bias_oscillator_invalid_length():
    high, low, close = mixed_data(64)
    with pytest.raises(ValueError, match="(?i)invalid length"):
        ta_indicators.geometric_bias_oscillator(
            high, low, close, length=9, multiplier=2.0, atr_length=14, smooth=1
        )


def test_geometric_bias_oscillator_stream_matches_batch():
    high, low, close = mixed_data(200)
    batch = ta_indicators.geometric_bias_oscillator(
        high, low, close, length=24, multiplier=2.4, atr_length=12, smooth=4
    )

    stream = ta_indicators.GeometricBiasOscillatorStream(
        length=24, multiplier=2.4, atr_length=12, smooth=4
    )
    streamed = []
    for h, l, c in zip(high, low, close):
        current = stream.update(float(h), float(l), float(c))
        streamed.append(np.nan if current is None else current)
    streamed = np.array(streamed, dtype=np.float64)

    assert_close(streamed, batch, atol=1e-12)


def test_geometric_bias_oscillator_batch_matches_single():
    high, low, close = mixed_data(170)
    batch = ta_indicators.geometric_bias_oscillator_batch(
        high,
        low,
        close,
        length_range=(10, 12, 2),
        multiplier_range=(1.5, 2.0, 0.5),
        atr_length_range=(5, 6, 1),
        smooth_range=(1, 2, 1),
    )

    assert batch["values"].shape == (16, close.size)
    assert batch["rows"] == 16
    assert batch["cols"] == close.size

    for row in range(batch["rows"]):
        single = ta_indicators.geometric_bias_oscillator(
            high,
            low,
            close,
            length=int(batch["lengths"][row]),
            multiplier=float(batch["multipliers"][row]),
            atr_length=int(batch["atr_lengths"][row]),
            smooth=int(batch["smooths"][row]),
        )
        assert_close(batch["values"][row], single, atol=1e-12)
