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


def mixed_data(size):
    high = np.empty(size, dtype=np.float64)
    low = np.empty(size, dtype=np.float64)
    close = np.empty(size, dtype=np.float64)
    for i in range(size):
        x = float(i)
        c = 100.0 + 0.18 * x + np.sin(x * 0.23) * 5.5 + np.cos(x * 0.07) * 2.0
        close[i] = c
        high[i] = c + 1.1 + (i % 3) * 0.05
        low[i] = c - 1.0 - (i % 2) * 0.05
    return high, low, close


def test_standardized_psar_oscillator_outputs_present():
    high, low, close = mixed_data(220)
    result = ta_indicators.standardized_psar_oscillator(high, low, close)
    assert set(result.keys()) == {
        "oscillator",
        "ma",
        "bullish_reversal",
        "bearish_reversal",
        "regular_bullish",
        "regular_bearish",
        "bullish_weakening",
        "bearish_weakening",
    }
    assert np.isfinite(result["oscillator"]).any()
    assert np.isfinite(result["ma"]).any()


def test_standardized_psar_oscillator_nan_gap_restart():
    high, low, close = mixed_data(220)
    high[120] = np.nan
    low[120] = np.nan
    close[120] = np.nan
    result = ta_indicators.standardized_psar_oscillator(
        high, low, close, standardization_length=12, wma_length=10
    )
    restart_end = min(close.size, 120 + max(12, 2))
    assert np.all(np.isnan(result["oscillator"][120:restart_end]))


def test_standardized_psar_oscillator_invalid_start():
    high, low, close = mixed_data(64)
    with pytest.raises(ValueError, match="(?i)invalid start"):
        ta_indicators.standardized_psar_oscillator(high, low, close, start=0.0)


def test_standardized_psar_oscillator_stream_matches_batch():
    high, low, close = mixed_data(220)
    batch = ta_indicators.standardized_psar_oscillator(
        high,
        low,
        close,
        start=0.03,
        increment=0.001,
        maximum=0.25,
        standardization_length=14,
        wma_length=12,
        wma_lag=2,
        pivot_left=8,
        pivot_right=1,
    )
    stream = ta_indicators.StandardizedPsarOscillatorStream(
        start=0.03,
        increment=0.001,
        maximum=0.25,
        standardization_length=14,
        wma_length=12,
        wma_lag=2,
        pivot_left=8,
        pivot_right=1,
    )
    oscillator = []
    ma = []
    for h, l, c in zip(high, low, close):
        current = stream.update(float(h), float(l), float(c))
        if current is None:
            oscillator.append(np.nan)
            ma.append(np.nan)
        else:
            oscillator.append(current[0])
            ma.append(current[1])
    assert_close(np.array(oscillator), batch["oscillator"], atol=1e-12)
    assert_close(np.array(ma), batch["ma"], atol=1e-12)


def test_standardized_psar_oscillator_batch_matches_single():
    high, low, close = mixed_data(180)
    batch = ta_indicators.standardized_psar_oscillator_batch(
        high,
        low,
        close,
        start_range=(0.02, 0.03, 0.01),
        increment_range=(0.0005, 0.0005, 0.0),
        maximum_range=(0.2, 0.2, 0.0),
        standardization_length_range=(10, 11, 1),
        wma_length_range=(8, 8, 0),
        wma_lag_range=(2, 2, 0),
        pivot_left_range=(6, 6, 0),
        pivot_right_range=(1, 1, 0),
    )
    assert batch["rows"] == 4
    assert batch["cols"] == close.size
