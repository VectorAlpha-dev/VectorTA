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


def sample_ohlcv(length=256):
    idx = np.arange(length, dtype=np.float64)
    open_ = 100.0 + idx * 0.03 + np.sin(idx * 0.08)
    close = open_ + np.cos(idx * 0.11) * 0.9
    high = np.maximum(open_, close) + 0.6 + np.abs(np.sin(idx * 0.03)) * 0.25
    low = np.minimum(open_, close) - 0.6 - np.abs(np.cos(idx * 0.05)) * 0.2
    volume = 1000.0 + idx * 4.0 + np.sin(idx * 0.09) * 180.0
    return high, low, close, volume


def test_volume_energy_reservoirs_output_contract():
    high, low, close, volume = sample_ohlcv()
    momentum, reservoir, squeeze_active, squeeze_start, range_high, range_low = (
        mp.volume_energy_reservoirs(high, low, close, volume)
    )

    assert len(momentum) == len(close)
    assert len(reservoir) == len(close)
    assert len(squeeze_active) == len(close)
    assert len(squeeze_start) == len(close)
    assert len(range_high) == len(close)
    assert len(range_low) == len(close)
    finite_reservoir = reservoir[np.isfinite(reservoir)]
    assert finite_reservoir.size > 0
    assert np.all(finite_reservoir >= -1e-12)
    assert np.all(finite_reservoir <= 10.0 + 1e-12)


def test_volume_energy_reservoirs_rejects_invalid_parameters():
    high, low, close, volume = sample_ohlcv(32)

    with pytest.raises(ValueError, match="Invalid length"):
        mp.volume_energy_reservoirs(high, low, close, volume, length=4)

    with pytest.raises(ValueError, match="Invalid sensitivity"):
        mp.volume_energy_reservoirs(high, low, close, volume, sensitivity=float("nan"))


def test_volume_energy_reservoirs_stream_matches_batch_with_reset():
    high, low, close, volume = sample_ohlcv(220)
    high[110] = np.nan
    low[110] = np.nan
    close[110] = np.nan
    volume[110] = np.nan

    batch = mp.volume_energy_reservoirs(high, low, close, volume, length=18, sensitivity=1.7)
    stream = mp.VolumeEnergyReservoirsStream(18, 1.7)

    streamed = [[] for _ in range(6)]
    for values in zip(high, low, close, volume):
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


def test_volume_energy_reservoirs_batch_single_param_matches_single():
    high, low, close, volume = sample_ohlcv(128)
    batch = mp.volume_energy_reservoirs_batch(
        high,
        low,
        close,
        volume,
        length_range=(18, 18, 0),
        sensitivity_range=(1.7, 1.7, 0.0),
    )
    single = mp.volume_energy_reservoirs(high, low, close, volume, length=18, sensitivity=1.7)

    assert batch["rows"] == 1
    assert batch["cols"] == len(close)
    np.testing.assert_array_equal(batch["lengths"], np.array([18], dtype=np.uint64))
    np.testing.assert_allclose(batch["sensitivities"], np.array([1.7]), atol=1e-12)

    keys = [
        "momentum",
        "reservoir",
        "squeeze_active",
        "squeeze_start",
        "range_high",
        "range_low",
    ]
    for key, expected in zip(keys, single):
        np.testing.assert_allclose(
            batch[key][0], expected, rtol=1e-12, atol=1e-12, equal_nan=True
        )


def test_volume_energy_reservoirs_batch_metadata():
    high, low, close, volume = sample_ohlcv(96)
    batch = mp.volume_energy_reservoirs_batch(
        high,
        low,
        close,
        volume,
        length_range=(16, 20, 4),
        sensitivity_range=(1.5, 2.0, 0.5),
    )

    assert batch["rows"] == 4
    assert batch["cols"] == len(close)
    assert batch["momentum"].shape == (4, len(close))
    np.testing.assert_array_equal(batch["lengths"], np.array([16, 16, 20, 20], dtype=np.uint64))
    np.testing.assert_allclose(batch["sensitivities"], np.array([1.5, 2.0, 1.5, 2.0]), atol=1e-12)
