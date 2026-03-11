import numpy as np
import pytest

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


def sample_ohlc(length=256):
    high = np.empty(length, dtype=np.float64)
    low = np.empty(length, dtype=np.float64)
    close = np.empty(length, dtype=np.float64)
    price = 100.0
    for i in range(length):
        wave = np.sin(i * 0.17) * 1.8 + np.cos(i * 0.07) * 0.9
        price += wave * 0.35 + 0.12
        close[i] = price
        high[i] = price + 0.8 + (i % 5) * 0.05
        low[i] = price - 0.7 - (i % 7) * 0.04
    return high, low, close


def test_kpo_output_contract_and_consistency():
    high, low, close = sample_ohlc()
    out = mp.kase_peak_oscillator_with_divergences(high, low, close)
    assert len(out) == 11
    oscillator = out[0]
    histogram = out[1]
    market_extreme = out[4]
    go_long = out[9]
    go_short = out[10]
    assert len(oscillator) == len(close)
    assert np.all(np.isnan(oscillator[:66]))
    np.testing.assert_allclose(oscillator, histogram, equal_nan=True)
    for i, value in enumerate(market_extreme):
        if np.isnan(value):
            continue
        if value < 0:
            assert go_long[i] == 1.0
        if value > 0:
            assert go_short[i] == 1.0


def test_kpo_kernel_parity():
    high, low, close = sample_ohlc(220)
    auto = mp.kase_peak_oscillator_with_divergences(high, low, close)
    scalar = mp.kase_peak_oscillator_with_divergences(
        high, low, close, kernel="scalar"
    )
    for lhs, rhs in zip(auto, scalar):
        np.testing.assert_allclose(lhs, rhs, rtol=1e-12, atol=1e-12, equal_nan=True)


def test_kpo_rejects_invalid_params():
    high, low, close = sample_ohlc(64)
    with pytest.raises(ValueError, match="Invalid cycle order"):
        mp.kase_peak_oscillator_with_divergences(
            high, low, close, short_cycle=10, long_cycle=10
        )
    with pytest.raises(ValueError, match="Invalid deviations"):
        mp.kase_peak_oscillator_with_divergences(
            high, low, close, deviations=-1.0
        )


def test_kpo_stream_matches_batch_with_reset():
    high, low, close = sample_ohlc(220)
    high[110] = np.nan
    low[110] = np.nan
    close[110] = np.nan
    batch = mp.kase_peak_oscillator_with_divergences(high, low, close)
    stream = mp.KasePeakOscillatorWithDivergencesStream()
    streamed = [[] for _ in range(11)]
    for h, l, c in zip(high, low, close):
        out = stream.update(float(h), float(l), float(c))
        if out is None:
            for series in streamed:
                series.append(np.nan)
        else:
            for series, value in zip(streamed, out):
                series.append(value)
    assert stream.warmup_period == 66
    for lhs, rhs in zip(batch, streamed):
        np.testing.assert_allclose(lhs, rhs, rtol=1e-12, atol=1e-12, equal_nan=True)


def test_kpo_batch_single_param_and_metadata():
    high, low, close = sample_ohlc(192)
    batch = mp.kase_peak_oscillator_with_divergences_batch(
        high,
        low,
        close,
        deviations_range=(2.0, 2.0, 0.0),
        short_cycle_range=(8, 8, 0),
        long_cycle_range=(65, 65, 0),
        sensitivity_range=(40.0, 40.0, 0.0),
    )
    single = mp.kase_peak_oscillator_with_divergences(high, low, close)
    assert batch["rows"] == 1
    assert batch["cols"] == len(close)
    np.testing.assert_array_equal(batch["deviations"], np.array([2.0]))
    np.testing.assert_array_equal(batch["short_cycles"], np.array([8], dtype=np.uint64))
    np.testing.assert_array_equal(batch["long_cycles"], np.array([65], dtype=np.uint64))
    np.testing.assert_array_equal(batch["sensitivities"], np.array([40.0]))
    np.testing.assert_allclose(batch["oscillator"][0], single[0], rtol=1e-12, atol=1e-12, equal_nan=True)
