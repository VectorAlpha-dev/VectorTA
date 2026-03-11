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
    open_ = np.array(
        [100.0 + i * 0.03 + np.sin(i * 0.09) for i in range(length)],
        dtype=np.float64,
    )
    close = np.array(
        [o + np.cos(i * 0.13) * 0.8 for i, o in enumerate(open_)],
        dtype=np.float64,
    )
    high = np.array(
        [max(o, c) + 0.5 + abs(np.sin(i * 0.05)) * 0.2 for i, (o, c) in enumerate(zip(open_, close))],
        dtype=np.float64,
    )
    low = np.array(
        [min(o, c) - 0.5 - abs(np.cos(i * 0.07)) * 0.2 for i, (o, c) in enumerate(zip(open_, close))],
        dtype=np.float64,
    )
    return open_, high, low, close


def test_fibonacci_trailing_stop_output_contract():
    _, high, low, close = sample_ohlc()
    trailing_stop, long_stop, short_stop, direction = mp.fibonacci_trailing_stop(high, low, close)

    assert len(trailing_stop) == len(close)
    assert len(long_stop) == len(close)
    assert len(short_stop) == len(close)
    assert len(direction) == len(close)
    assert np.isfinite(trailing_stop[0])
    finite_direction = direction[np.isfinite(direction)]
    assert finite_direction.size > 0
    assert np.all(np.isin(finite_direction, np.array([-1.0, 0.0, 1.0], dtype=np.float64)))


def test_fibonacci_trailing_stop_rejects_invalid_parameters():
    _, high, low, close = sample_ohlc(32)

    with pytest.raises(ValueError, match="Invalid left_bars"):
        mp.fibonacci_trailing_stop(high, low, close, left_bars=0)

    with pytest.raises(ValueError, match="Invalid trigger"):
        mp.fibonacci_trailing_stop(high, low, close, trigger="bad")


def test_fibonacci_trailing_stop_stream_matches_batch_with_reset():
    _, high, low, close = sample_ohlc(220)
    high[110] = np.nan
    low[110] = np.nan
    close[110] = np.nan

    batch = mp.fibonacci_trailing_stop(high, low, close, left_bars=12, right_bars=2, level=-0.236)
    stream = mp.FibonacciTrailingStopStream(12, 2, -0.236, "close")

    streamed = [[], [], [], []]
    for h, l, c in zip(high, low, close):
        out = stream.update(float(h), float(l), float(c))
        if out is None:
            for series in streamed:
                series.append(np.nan)
        else:
            for series, value in zip(streamed, out):
                series.append(value)

    for batch_series, stream_series in zip(batch, streamed):
        np.testing.assert_allclose(
            batch_series,
            stream_series,
            rtol=1e-12,
            atol=1e-12,
            equal_nan=True,
        )


def test_fibonacci_trailing_stop_batch_single_param_matches_single():
    _, high, low, close = sample_ohlc(160)
    batch = mp.fibonacci_trailing_stop_batch(
        high,
        low,
        close,
        left_bars_range=(12, 12, 0),
        right_bars_range=(2, 2, 0),
        level_range=(-0.236, -0.236, 0.0),
        trigger="wick",
    )
    single = mp.fibonacci_trailing_stop(
        high,
        low,
        close,
        left_bars=12,
        right_bars=2,
        level=-0.236,
        trigger="wick",
    )

    assert batch["rows"] == 1
    assert batch["cols"] == len(close)
    np.testing.assert_array_equal(batch["left_bars"], np.array([12], dtype=np.uint64))
    np.testing.assert_array_equal(batch["right_bars"], np.array([2], dtype=np.uint64))
    np.testing.assert_allclose(batch["levels"], np.array([-0.236], dtype=np.float64))
    np.testing.assert_allclose(
        batch["trailing_stop"][0], single[0], rtol=1e-12, atol=1e-12, equal_nan=True
    )
    np.testing.assert_allclose(
        batch["long_stop"][0], single[1], rtol=1e-12, atol=1e-12, equal_nan=True
    )
    np.testing.assert_allclose(
        batch["short_stop"][0], single[2], rtol=1e-12, atol=1e-12, equal_nan=True
    )
    np.testing.assert_allclose(
        batch["direction"][0], single[3], rtol=1e-12, atol=1e-12, equal_nan=True
    )


def test_fibonacci_trailing_stop_batch_metadata():
    _, high, low, close = sample_ohlc(120)
    batch = mp.fibonacci_trailing_stop_batch(
        high,
        low,
        close,
        left_bars_range=(10, 12, 2),
        right_bars_range=(1, 2, 1),
        level_range=(-0.382, -0.236, 0.146),
    )

    assert batch["rows"] == 8
    assert batch["cols"] == len(close)
    assert batch["trailing_stop"].shape == (8, len(close))
    assert batch["long_stop"].shape == (8, len(close))
    assert batch["short_stop"].shape == (8, len(close))
    assert batch["direction"].shape == (8, len(close))
    np.testing.assert_array_equal(
        batch["left_bars"], np.array([10, 10, 10, 10, 12, 12, 12, 12], dtype=np.uint64)
    )
    np.testing.assert_array_equal(
        batch["right_bars"], np.array([1, 1, 2, 2, 1, 1, 2, 2], dtype=np.uint64)
    )
    np.testing.assert_allclose(
        batch["levels"], np.array([-0.382, -0.236, -0.382, -0.236, -0.382, -0.236, -0.382, -0.236])
    )
