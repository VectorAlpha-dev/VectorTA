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


def trend_data(size):
    high = np.empty(size, dtype=np.float64)
    low = np.empty(size, dtype=np.float64)
    close = np.empty(size, dtype=np.float64)
    for i in range(size):
        base = 100.0 + i * 0.8
        high[i] = base + 1.2 + (i % 3) * 0.1
        low[i] = base - 1.0 - (i % 2) * 0.1
        close[i] = base + ((i % 5) - 2) * 0.05
    return high, low, close


def reversal_data():
    close = np.array(
        [
            100.0,
            101.0,
            102.0,
            103.0,
            104.0,
            105.0,
            104.0,
            102.0,
            99.0,
            96.0,
            93.0,
            91.0,
            90.0,
            91.0,
            93.0,
            96.0,
            100.0,
            105.0,
            109.0,
            112.0,
            111.0,
            109.0,
            107.0,
            104.0,
            101.0,
            98.0,
            96.0,
            95.0,
            96.0,
            98.0,
        ],
        dtype=np.float64,
    )
    return close + 1.0, close - 1.0, close


def recovery_data():
    close = np.array(
        [
            100.0,
            101.0,
            102.0,
            103.0,
            104.0,
            105.0,
            102.0,
            97.0,
            92.0,
            88.0,
            90.0,
            93.0,
            96.0,
            99.0,
            101.0,
            102.0,
            101.0,
            100.0,
            99.0,
            98.0,
            97.0,
            96.0,
            95.0,
            94.0,
            93.0,
            92.0,
            91.0,
            90.0,
            89.0,
            88.0,
        ],
        dtype=np.float64,
    )
    return close + 0.9, close - 0.9, close


def test_supertrend_recovery_reversal_behavior():
    high, low, close = reversal_data()
    band, switch_price, trend, changed = ta_indicators.supertrend_recovery(
        high, low, close, atr_length=4, multiplier=1.5, alpha_percent=5.0, threshold_atr=1.0
    )

    changed_idx = np.flatnonzero(changed == 1.0)
    assert changed_idx.size > 0
    first = changed_idx[0]
    assert np.isfinite(band[first])
    assert np.isfinite(switch_price[first])
    assert trend[first] in (-1.0, 1.0)


def test_supertrend_recovery_invalid_alpha():
    high, low, close = trend_data(160)
    with pytest.raises(ValueError, match="(?i)invalid alpha_percent"):
        ta_indicators.supertrend_recovery(
            high, low, close, atr_length=10, multiplier=3.0, alpha_percent=0.0, threshold_atr=1.0
        )


def test_supertrend_recovery_nan_gap_restart():
    high, low, close = trend_data(170)
    high[120] = np.nan
    low[120] = np.nan
    close[120] = np.nan
    band, switch_price, trend, changed = ta_indicators.supertrend_recovery(
        high, low, close, atr_length=10, multiplier=3.0, alpha_percent=5.0, threshold_atr=1.0
    )

    restart_end = 120 + 10
    assert np.all(np.isnan(band[120:restart_end]))
    assert np.all(np.isnan(trend[120:restart_end]))
    assert np.all(np.isnan(changed[120:restart_end]))
    assert np.all(np.isnan(switch_price[120:restart_end]))


def test_supertrend_recovery_recovery_logic_changes_band():
    high, low, close = recovery_data()
    recovered = ta_indicators.supertrend_recovery(
        high, low, close, atr_length=4, multiplier=3.0, alpha_percent=100.0, threshold_atr=0.0
    )
    baseline = ta_indicators.supertrend_recovery(
        high, low, close, atr_length=4, multiplier=3.0, alpha_percent=0.1, threshold_atr=1000.0
    )

    recovered_band, _, recovered_trend, _ = recovered
    baseline_band, _, baseline_trend, _ = baseline
    same_trend = recovered_trend == baseline_trend
    finite = np.isfinite(recovered_band) & np.isfinite(baseline_band)
    bullish = finite & same_trend & (recovered_trend == 1.0)
    bearish = finite & same_trend & (recovered_trend == -1.0)
    assert np.any(recovered_band[bullish] > baseline_band[bullish]) or np.any(
        recovered_band[bearish] < baseline_band[bearish]
    )


def test_supertrend_recovery_stream_matches_batch():
    high, low, close = trend_data(170)
    batch_band, batch_switch, batch_trend, batch_changed = ta_indicators.supertrend_recovery(
        high, low, close, atr_length=10, multiplier=3.0, alpha_percent=5.0, threshold_atr=1.0
    )

    stream = ta_indicators.SuperTrendRecoveryStream(
        atr_length=10, multiplier=3.0, alpha_percent=5.0, threshold_atr=1.0
    )
    band = []
    switch_price = []
    trend = []
    changed = []
    for h, l, c in zip(high, low, close):
        current = stream.update(float(h), float(l), float(c))
        if current is None:
            band.append(np.nan)
            switch_price.append(np.nan)
            trend.append(np.nan)
            changed.append(np.nan)
        else:
            b, s, t, ch = current
            band.append(b)
            switch_price.append(s)
            trend.append(t)
            changed.append(ch)

    assert_close(np.array(band), batch_band, atol=1e-12)
    assert_close(np.array(switch_price), batch_switch, atol=1e-12)
    assert_close(np.array(trend), batch_trend, atol=1e-12)
    assert_close(np.array(changed), batch_changed, atol=1e-12)


def test_supertrend_recovery_batch_matches_single():
    high, low, close = trend_data(170)
    batch = ta_indicators.supertrend_recovery_batch(
        high,
        low,
        close,
        atr_length_range=(4, 5, 1),
        multiplier_range=(1.5, 2.0, 0.5),
        alpha_percent_range=(5.0, 10.0, 5.0),
        threshold_atr_range=(0.5, 1.0, 0.5),
    )

    assert batch["band"].shape == (16, close.size)
    assert batch["switch_price"].shape == (16, close.size)
    assert batch["trend"].shape == (16, close.size)
    assert batch["changed"].shape == (16, close.size)
    assert batch["rows"] == 16
    assert batch["cols"] == close.size

    for row in range(batch["rows"]):
        band, switch_price, trend, changed = ta_indicators.supertrend_recovery(
            high,
            low,
            close,
            atr_length=int(batch["atr_lengths"][row]),
            multiplier=float(batch["multipliers"][row]),
            alpha_percent=float(batch["alpha_percents"][row]),
            threshold_atr=float(batch["threshold_atrs"][row]),
        )
        assert_close(batch["band"][row], band, atol=1e-12)
        assert_close(batch["switch_price"][row], switch_price, atol=1e-12)
        assert_close(batch["trend"][row], trend, atol=1e-12)
        assert_close(batch["changed"][row], changed, atol=1e-12)
