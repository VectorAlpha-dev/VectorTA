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


def test_hull_butterfly_oscillator_output_contract():
    data = np.array(
        [100.0 + i * 0.05 + np.sin(i * 0.17) * 2.4 + np.cos(i * 0.03) * 0.7 for i in range(256)],
        dtype=np.float64,
    )
    oscillator, cumulative_mean, signal = mp.hull_butterfly_oscillator(data)

    assert len(oscillator) == len(data)
    assert len(cumulative_mean) == len(data)
    assert len(signal) == len(data)
    finite_idx = np.flatnonzero(np.isfinite(oscillator))
    assert finite_idx.size > 0
    assert finite_idx[0] >= 15
    assert np.all(np.isfinite(cumulative_mean[finite_idx]))
    finite_signal = signal[np.isfinite(signal)]
    assert np.all(np.isin(finite_signal, np.array([-1.0, 0.0, 1.0], dtype=np.float64)))


def test_hull_butterfly_oscillator_kernel_parity():
    data = np.array(
        [100.0 + i * 0.05 + np.sin(i * 0.17) * 2.4 + np.cos(i * 0.03) * 0.7 for i in range(256)],
        dtype=np.float64,
    )
    auto = mp.hull_butterfly_oscillator(data, length=12, mult=1.75)
    scalar = mp.hull_butterfly_oscillator(data, length=12, mult=1.75, kernel="scalar")
    for auto_series, scalar_series in zip(auto, scalar):
        np.testing.assert_allclose(
            auto_series,
            scalar_series,
            rtol=1e-12,
            atol=1e-12,
            equal_nan=True,
        )


def test_hull_butterfly_oscillator_rejects_invalid_parameters():
    data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)

    with pytest.raises(ValueError, match="Invalid length"):
        mp.hull_butterfly_oscillator(data, length=1)

    with pytest.raises(ValueError, match="Invalid multiplier"):
        mp.hull_butterfly_oscillator(data, mult=float("nan"))


def test_hull_butterfly_oscillator_stream_matches_batch_with_reset():
    data = np.array(
        [100.0 + i * 0.05 + np.sin(i * 0.17) * 2.4 + np.cos(i * 0.03) * 0.7 for i in range(220)],
        dtype=np.float64,
    )
    data[110] = np.nan
    batch = mp.hull_butterfly_oscillator(data, length=12, mult=1.75)
    stream = mp.HullButterflyOscillatorStream(12, 1.75)

    streamed = [[], [], []]
    for value in data:
        out = stream.update(float(value))
        if out is None:
            streamed[0].append(np.nan)
            streamed[1].append(np.nan)
            streamed[2].append(np.nan)
        else:
            streamed[0].append(out[0])
            streamed[1].append(out[1])
            streamed[2].append(out[2])

    assert stream.warmup_period == 13
    for batch_series, stream_series in zip(batch, streamed):
        np.testing.assert_allclose(
            batch_series,
            stream_series,
            rtol=1e-12,
            atol=1e-12,
            equal_nan=True,
        )


def test_hull_butterfly_oscillator_batch_single_param_matches_single():
    data = np.array(
        [100.0 + i * 0.05 + np.sin(i * 0.17) * 2.4 + np.cos(i * 0.03) * 0.7 for i in range(160)],
        dtype=np.float64,
    )
    batch = mp.hull_butterfly_oscillator_batch(
        data,
        length_range=(12, 12, 0),
        mult_range=(1.5, 1.5, 0.0),
    )
    single = mp.hull_butterfly_oscillator(data, length=12, mult=1.5)

    assert batch["rows"] == 1
    assert batch["cols"] == len(data)
    np.testing.assert_array_equal(batch["lengths"], np.array([12], dtype=np.uint64))
    np.testing.assert_allclose(batch["multipliers"], np.array([1.5], dtype=np.float64))
    np.testing.assert_allclose(
        batch["oscillator"][0], single[0], rtol=1e-12, atol=1e-12, equal_nan=True
    )
    np.testing.assert_allclose(
        batch["cumulative_mean"][0], single[1], rtol=1e-12, atol=1e-12, equal_nan=True
    )
    np.testing.assert_allclose(
        batch["signal"][0], single[2], rtol=1e-12, atol=1e-12, equal_nan=True
    )


def test_hull_butterfly_oscillator_batch_metadata():
    data = np.array(
        [100.0 + i * 0.05 + np.sin(i * 0.17) * 2.4 + np.cos(i * 0.03) * 0.7 for i in range(120)],
        dtype=np.float64,
    )
    batch = mp.hull_butterfly_oscillator_batch(
        data,
        length_range=(10, 14, 2),
        mult_range=(1.0, 2.0, 0.5),
    )

    assert batch["rows"] == 9
    assert batch["cols"] == len(data)
    assert batch["oscillator"].shape == (9, len(data))
    assert batch["cumulative_mean"].shape == (9, len(data))
    assert batch["signal"].shape == (9, len(data))
    np.testing.assert_array_equal(
        batch["lengths"], np.array([10, 10, 10, 12, 12, 12, 14, 14, 14], dtype=np.uint64)
    )
    np.testing.assert_allclose(
        batch["multipliers"], np.array([1.0, 1.5, 2.0, 1.0, 1.5, 2.0, 1.0, 1.5, 2.0])
    )
