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


def sample_ohlc(length: int):
    open_ = np.full(length, np.nan, dtype=np.float64)
    high = np.full(length, np.nan, dtype=np.float64)
    low = np.full(length, np.nan, dtype=np.float64)
    close = np.full(length, np.nan, dtype=np.float64)
    prev = 100.0
    for i in range(2, length):
        x = float(i)
        wave = np.sin(x * 0.11) * 2.4 + np.cos(x * 0.037) * 1.3
        o = prev + wave * 0.35
        c = o + np.sin(x * 0.19) * 1.1 - np.cos(x * 0.07) * 0.4
        h = max(o, c) + 0.6 + abs(np.sin(x * 0.03)) * 0.25
        l = min(o, c) - 0.6 - abs(np.cos(x * 0.02)) * 0.25
        open_[i] = o
        high[i] = h
        low[i] = l
        close[i] = c
        prev = c
    return open_, high, low, close


def test_grover_llorens_cycle_oscillator_output_contract():
    open_, high, low, close = sample_ohlc(512)
    values = mp.grover_llorens_cycle_oscillator(
        open_, high, low, close, length=100, mult=10.0, source="close", smooth=True, rsi_period=20
    )

    assert len(values) == len(close)
    valid = np.flatnonzero(~np.isnan(values))
    assert valid.size > 0
    assert int(valid[0]) >= 100
    tail_start = min(int(valid[0]) + 32, len(values))
    assert not np.any(np.isnan(values[tail_start:]))


def test_grover_llorens_cycle_oscillator_rejects_invalid_parameters():
    open_, high, low, close = sample_ohlc(64)

    with pytest.raises(ValueError, match="Invalid length"):
        mp.grover_llorens_cycle_oscillator(
            open_, high, low, close, length=0, mult=10.0, source="close", smooth=True, rsi_period=20
        )

    with pytest.raises(ValueError, match="Invalid source"):
        mp.grover_llorens_cycle_oscillator(
            open_, high, low, close, length=10, mult=10.0, source="bad", smooth=True, rsi_period=20
        )

    with pytest.raises(ValueError, match="length mismatch"):
        mp.grover_llorens_cycle_oscillator(
            open_[:-1], high, low, close, length=10, mult=10.0, source="close", smooth=True, rsi_period=20
        )


def test_grover_llorens_cycle_oscillator_stream_matches_batch_with_reset():
    open_, high, low, close = sample_ohlc(256)
    open_[120] = np.nan
    high[120] = np.nan
    low[120] = np.nan
    close[120] = np.nan

    batch = mp.grover_llorens_cycle_oscillator(
        open_, high, low, close, length=60, mult=8.0, source="hlc3", smooth=True, rsi_period=14
    )
    stream = mp.GroverLlorensCycleOscillatorStream(60, 8.0, "hlc3", True, 14)

    streamed = []
    for o, h, l, c in zip(open_, high, low, close):
        out = stream.update(float(o), float(h), float(l), float(c))
        streamed.append(np.nan if out is None else out)

    np.testing.assert_allclose(
        np.asarray(streamed), batch, rtol=1e-12, atol=1e-12, equal_nan=True
    )


def test_grover_llorens_cycle_oscillator_batch_single_param_matches_single():
    open_, high, low, close = sample_ohlc(192)
    batch = mp.grover_llorens_cycle_oscillator_batch(
        open_,
        high,
        low,
        close,
        length_range=(50, 50, 0),
        mult_range=(8.0, 8.0, 0.0),
        source="close",
        smooth=True,
        rsi_period_range=(14, 14, 0),
    )
    single = mp.grover_llorens_cycle_oscillator(
        open_, high, low, close, length=50, mult=8.0, source="close", smooth=True, rsi_period=14
    )

    assert batch["rows"] == 1
    assert batch["cols"] == len(close)
    np.testing.assert_array_equal(batch["lengths"], np.array([50], dtype=np.uint64))
    np.testing.assert_allclose(batch["mults"], np.array([8.0]), rtol=0.0, atol=1e-12)
    assert batch["sources"] == ["close"]
    np.testing.assert_array_equal(batch["smooth_flags"], np.array([True]))
    np.testing.assert_array_equal(batch["rsi_periods"], np.array([14], dtype=np.uint64))
    np.testing.assert_allclose(
        batch["values"][0], single, rtol=1e-12, atol=1e-12, equal_nan=True
    )


def test_grover_llorens_cycle_oscillator_batch_metadata():
    open_, high, low, close = sample_ohlc(160)
    batch = mp.grover_llorens_cycle_oscillator_batch(
        open_,
        high,
        low,
        close,
        length_range=(40, 60, 20),
        mult_range=(8.0, 10.0, 2.0),
        source="hlc3",
        smooth=False,
        rsi_period_range=(14, 16, 2),
    )

    assert batch["rows"] == 8
    assert batch["cols"] == len(close)
    assert batch["values"].shape == (8, len(close))
    np.testing.assert_array_equal(batch["lengths"], np.array([40, 40, 40, 40, 60, 60, 60, 60], dtype=np.uint64))
    np.testing.assert_allclose(batch["mults"], np.array([8.0, 8.0, 10.0, 10.0, 8.0, 8.0, 10.0, 10.0]), rtol=0.0, atol=1e-12)
    assert batch["sources"] == ["hlc3"] * 8
    np.testing.assert_array_equal(batch["smooth_flags"], np.array([False] * 8))
    np.testing.assert_array_equal(batch["rsi_periods"], np.array([14, 16, 14, 16, 14, 16, 14, 16], dtype=np.uint64))
