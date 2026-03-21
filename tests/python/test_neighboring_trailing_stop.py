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


def sample_ohlc(length=256):
    idx = np.arange(length, dtype=np.float64)
    open_ = 100.0 + idx * 0.04 + np.sin(idx * 0.07)
    close = open_ + np.cos(idx * 0.11) * 0.85
    high = np.maximum(open_, close) + 0.55 + np.abs(np.sin(idx * 0.03)) * 0.2
    low = np.minimum(open_, close) - 0.55 - np.abs(np.cos(idx * 0.05)) * 0.2
    return high, low, close


def test_neighboring_trailing_stop_output_contract():
    high, low, close = sample_ohlc()
    trailing_stop, bullish_band, bearish_band, direction, discovery_bull, discovery_bear = (
        mp.neighboring_trailing_stop(high, low, close)
    )

    assert len(trailing_stop) == len(close)
    assert len(bullish_band) == len(close)
    assert len(bearish_band) == len(close)
    assert len(direction) == len(close)
    assert len(discovery_bull) == len(close)
    assert len(discovery_bear) == len(close)

    finite_direction = direction[np.isfinite(direction)]
    assert finite_direction.size > 0
    assert np.all(np.isin(finite_direction, np.array([-1.0, 0.0, 1.0])))


def test_neighboring_trailing_stop_rejects_invalid_parameters():
    high, low, close = sample_ohlc(64)

    with pytest.raises(ValueError, match="Invalid buffer_size"):
        mp.neighboring_trailing_stop(high, low, close, buffer_size=50)

    with pytest.raises(ValueError, match="Invalid k"):
        mp.neighboring_trailing_stop(high, low, close, k=2)

    with pytest.raises(ValueError, match="Invalid percentile"):
        mp.neighboring_trailing_stop(high, low, close, percentile=float("nan"))


def test_neighboring_trailing_stop_stream_matches_batch_with_reset():
    high, low, close = sample_ohlc(220)
    high[110] = np.nan
    low[110] = np.nan
    close[110] = np.nan

    batch = mp.neighboring_trailing_stop(
        high, low, close, buffer_size=180, k=25, percentile=92.0, smooth=4
    )
    stream = mp.NeighboringTrailingStopStream(180, 25, 92.0, 4)

    streamed = [[] for _ in range(6)]
    for values in zip(high, low, close):
        out = stream.update(*map(float, values))
        if out is None:
            for series in streamed:
                series.append(np.nan)
        else:
            for series, value in zip(streamed, out):
                series.append(value)

    assert stream.warmup_period == 0
    for batch_series, stream_series in zip(batch, streamed):
        np.testing.assert_allclose(
            batch_series, stream_series, rtol=1e-12, atol=1e-12, equal_nan=True
        )


def test_neighboring_trailing_stop_batch_single_param_matches_single():
    high, low, close = sample_ohlc(128)
    batch = mp.neighboring_trailing_stop_batch(
        high,
        low,
        close,
        buffer_size_range=(180, 180, 0),
        k_range=(30, 30, 0),
        percentile_range=(87.5, 87.5, 0.0),
        smooth_range=(4, 4, 0),
    )
    single = mp.neighboring_trailing_stop(
        high, low, close, buffer_size=180, k=30, percentile=87.5, smooth=4
    )

    assert batch["rows"] == 1
    assert batch["cols"] == len(close)
    np.testing.assert_array_equal(batch["buffer_sizes"], np.array([180], dtype=np.uint64))
    np.testing.assert_array_equal(batch["ks"], np.array([30], dtype=np.uint64))
    np.testing.assert_allclose(batch["percentiles"], np.array([87.5]), atol=1e-12)
    np.testing.assert_array_equal(batch["smoothings"], np.array([4], dtype=np.uint64))

    keys = [
        "trailing_stop",
        "bullish_band",
        "bearish_band",
        "direction",
        "discovery_bull",
        "discovery_bear",
    ]
    for key, expected in zip(keys, single):
        np.testing.assert_allclose(
            batch[key][0], expected, rtol=1e-12, atol=1e-12, equal_nan=True
        )


def test_neighboring_trailing_stop_batch_metadata():
    high, low, close = sample_ohlc(96)
    batch = mp.neighboring_trailing_stop_batch(
        high,
        low,
        close,
        buffer_size_range=(150, 150, 0),
        k_range=(20, 24, 4),
        percentile_range=(85.0, 90.0, 5.0),
        smooth_range=(3, 3, 0),
    )

    assert batch["rows"] == 4
    assert batch["cols"] == len(close)
    assert batch["trailing_stop"].shape == (4, len(close))
    np.testing.assert_array_equal(batch["buffer_sizes"], np.array([150, 150, 150, 150], dtype=np.uint64))
    np.testing.assert_array_equal(batch["ks"], np.array([20, 20, 24, 24], dtype=np.uint64))
    np.testing.assert_allclose(batch["percentiles"], np.array([85.0, 90.0, 85.0, 90.0]), atol=1e-12)
    np.testing.assert_array_equal(batch["smoothings"], np.array([3, 3, 3, 3], dtype=np.uint64))
