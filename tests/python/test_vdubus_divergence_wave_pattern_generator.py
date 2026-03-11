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

from test_utils import assert_close


def oscillating_data(size):
    high = np.empty(size, dtype=np.float64)
    low = np.empty(size, dtype=np.float64)
    close = np.empty(size, dtype=np.float64)
    for i in range(size):
        x = float(i)
        c = 100.0 + np.sin(x * 0.19) * 12.0 + np.cos(x * 0.043) * 5.0 + 0.03 * x
        close[i] = c
        high[i] = c + 1.4 + (i % 5) * 0.07
        low[i] = c - 1.3 - (i % 3) * 0.05
    return high, low, close


def test_vdubus_generator_outputs_hist_and_force():
    high, low, close = oscillating_data(320)
    result = ta_indicators.vdubus_divergence_wave_pattern_generator(high, low, close)
    assert np.isfinite(result["hist"]).any()
    assert np.isfinite(result["opposing_force"]).any()


def test_vdubus_generator_nan_gap_restart():
    high, low, close = oscillating_data(260)
    high[140] = np.nan
    low[140] = np.nan
    close[140] = np.nan
    result = ta_indicators.vdubus_divergence_wave_pattern_generator(high, low, close)
    restart_end = min(result["hist"].size, 140 + 34 + 5 - 1)
    assert np.all(np.isnan(result["hist"][140:restart_end]))


def test_vdubus_generator_invalid_period():
    high, low, close = oscillating_data(64)
    with pytest.raises(ValueError, match="(?i)invalid periods"):
        ta_indicators.vdubus_divergence_wave_pattern_generator(
            high, low, close, slow_length=0
        )


def test_vdubus_generator_stream_matches_single_hist():
    high, low, close = oscillating_data(240)
    batch = ta_indicators.vdubus_divergence_wave_pattern_generator(high, low, close)
    stream = ta_indicators.VdubusDivergenceWavePatternGeneratorStream()
    hist = []
    for h, l, c in zip(high, low, close):
        current = stream.update(float(h), float(l), float(c))
        hist.append(np.nan if current is None else current[11])
    assert_close(np.array(hist, dtype=np.float64), batch["hist"], atol=1e-12)


def test_vdubus_generator_batch_matches_single_hist():
    high, low, close = oscillating_data(220)
    batch = ta_indicators.vdubus_divergence_wave_pattern_generator_batch(
        high,
        low,
        close,
        fast_depth_range=(7, 9, 2),
        slow_depth_range=(20, 24, 4),
        fast_length_range=(15, 17, 2),
        slow_length_range=(28, 30, 2),
        signal_length_range=(4, 5, 1),
        lookback_range=(2, 3, 1),
        err_tol_range=(0.10, 0.15, 0.05),
    )
    assert batch["rows"] == 128
    assert batch["cols"] == close.size
    single = ta_indicators.vdubus_divergence_wave_pattern_generator(
        high,
        low,
        close,
        fast_depth=7,
        slow_depth=20,
        fast_length=15,
        slow_length=28,
        signal_length=4,
        lookback=2,
        err_tol=0.10,
    )
    assert_close(batch["hist"][0], single["hist"], atol=1e-12)
