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


def trend_data(size):
    high = np.empty(size, dtype=np.float64)
    low = np.empty(size, dtype=np.float64)
    close = np.empty(size, dtype=np.float64)
    for i in range(size):
        base = 100.0 + i * 0.9
        high[i] = base + 1.0 + (i % 3) * 0.1
        low[i] = base - 1.0 - (i % 2) * 0.1
        close[i] = base
    return high, low, close


def reversal_data():
    close = np.array(
        [
            100.0,
            99.5,
            99.0,
            98.5,
            98.0,
            97.5,
            97.0,
            96.5,
            96.0,
            95.5,
            97.0,
            99.0,
            101.0,
            104.0,
            108.0,
            111.0,
            113.0,
            112.0,
            111.0,
            109.0,
            107.0,
            105.0,
            103.0,
            101.0,
            99.0,
            97.0,
            95.0,
            94.0,
            93.5,
            93.0,
        ],
        dtype=np.float64,
    )
    high = close + 0.8
    low = close - 0.8
    return high, low, close


def test_statistical_trailing_stop_reversal_behavior():
    high, low, close = reversal_data()
    level, anchor, bias, changed = ta_indicators.statistical_trailing_stop(
        high,
        low,
        close,
        data_length=3,
        normalization_length=10,
        base_level="level0",
    )

    changed_idx = np.flatnonzero(changed == 1.0)
    assert changed_idx.size > 0
    first = changed_idx[0]
    assert np.isfinite(anchor[first])
    assert bias[first] == 1.0


def test_statistical_trailing_stop_invalid_base_level():
    high, low, close = trend_data(160)
    with pytest.raises(ValueError, match="(?i)invalid base level"):
        ta_indicators.statistical_trailing_stop(
            high,
            low,
            close,
            data_length=10,
            normalization_length=100,
            base_level="bad",
        )


def test_statistical_trailing_stop_nan_gap_restart():
    high, low, close = trend_data(170)
    high[120] = np.nan
    low[120] = np.nan
    close[120] = np.nan
    level, anchor, bias, changed = ta_indicators.statistical_trailing_stop(
        high, low, close, data_length=10, normalization_length=100, base_level="level2"
    )

    restart_end = 120 + 10 + 100 + 1
    assert np.all(np.isnan(level[120:restart_end]))
    assert np.all(np.isnan(bias[120:restart_end]))
    assert np.all(np.isnan(changed[120:restart_end]))
    assert np.all(np.isnan(anchor[:restart_end]))


def test_statistical_trailing_stop_stream_matches_batch():
    high, low, close = trend_data(170)
    batch_level, batch_anchor, batch_bias, batch_changed = (
        ta_indicators.statistical_trailing_stop(
            high, low, close, data_length=10, normalization_length=100, base_level="level2"
        )
    )

    stream = ta_indicators.StatisticalTrailingStopStream(
        data_length=10, normalization_length=100, base_level="level2"
    )
    level = []
    anchor = []
    bias = []
    changed = []

    for h, l, c in zip(high, low, close):
        current = stream.update(float(h), float(l), float(c))
        if current is None:
            level.append(np.nan)
            anchor.append(np.nan)
            bias.append(np.nan)
            changed.append(np.nan)
        else:
            lvl, anc, bs, ch = current
            level.append(lvl)
            anchor.append(anc)
            bias.append(bs)
            changed.append(ch)

    assert_close(np.array(level), batch_level, atol=1e-12)
    assert_close(np.array(anchor), batch_anchor, atol=1e-12)
    assert_close(np.array(bias), batch_bias, atol=1e-12)
    assert_close(np.array(changed), batch_changed, atol=1e-12)


def test_statistical_trailing_stop_batch_matches_single():
    high, low, close = trend_data(170)
    batch = ta_indicators.statistical_trailing_stop_batch(
        high,
        low,
        close,
        data_length_range=(9, 10, 1),
        normalization_length_range=(10, 11, 1),
        base_level_range=("level1", "level2", 1),
    )

    assert batch["level"].shape == (8, close.size)
    assert batch["anchor"].shape == (8, close.size)
    assert batch["bias"].shape == (8, close.size)
    assert batch["changed"].shape == (8, close.size)
    assert batch["rows"] == 8
    assert batch["cols"] == close.size

    for row in range(batch["rows"]):
        level, anchor, bias, changed = ta_indicators.statistical_trailing_stop(
            high,
            low,
            close,
            data_length=int(batch["data_lengths"][row]),
            normalization_length=int(batch["normalization_lengths"][row]),
            base_level=batch["base_levels"][row],
        )
        assert_close(batch["level"][row], level, atol=1e-12)
        assert_close(batch["anchor"][row], anchor, atol=1e-12)
        assert_close(batch["bias"][row], bias, atol=1e-12)
        assert_close(batch["changed"][row], changed, atol=1e-12)
