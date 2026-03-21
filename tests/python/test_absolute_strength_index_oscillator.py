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


def test_absolute_strength_index_oscillator_output_contract_and_manual_values():
    data = np.array([10.0, 10.0], dtype=np.float64)
    oscillator, signal, histogram = mp.absolute_strength_index_oscillator(
        data, ema_length=2, signal_length=2
    )

    assert len(oscillator) == len(data)
    np.testing.assert_allclose(oscillator[0], 0.0, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(signal[0], 0.0, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(histogram[0], 0.0, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(oscillator[1], -1.0 / 6.0, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(signal[1], -2.0 / 9.0, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(histogram[1], 1.0 / 18.0, rtol=0.0, atol=1e-12)


def test_absolute_strength_index_oscillator_kernel_parity():
    data = np.linspace(90.0, 110.0, 256, dtype=np.float64)
    auto = mp.absolute_strength_index_oscillator(data, ema_length=21, signal_length=34)
    scalar = mp.absolute_strength_index_oscillator(
        data, ema_length=21, signal_length=34, kernel="scalar"
    )

    for lhs, rhs in zip(auto, scalar):
        np.testing.assert_allclose(lhs, rhs, rtol=1e-12, atol=1e-12, equal_nan=True)


def test_absolute_strength_index_oscillator_rejects_invalid_parameters():
    data = np.array([1.0, 2.0, 3.0], dtype=np.float64)

    with pytest.raises(ValueError, match="Invalid ema_length"):
        mp.absolute_strength_index_oscillator(data, ema_length=0, signal_length=2)

    with pytest.raises(ValueError, match="Invalid signal_length"):
        mp.absolute_strength_index_oscillator(data, ema_length=2, signal_length=1)


def test_absolute_strength_index_oscillator_stream_matches_batch_with_reset():
    data = np.array([10.0, 10.0, 9.0, np.nan, 10.0, 10.0], dtype=np.float64)
    batch = mp.absolute_strength_index_oscillator(data, ema_length=2, signal_length=2)
    stream = mp.AbsoluteStrengthIndexOscillatorStream(2, 2)

    streamed = [[], [], []]
    for value in data:
        out = stream.update(float(value))
        if out is None:
            for series in streamed:
                series.append(np.nan)
        else:
            for series, component in zip(streamed, out):
                series.append(component)

    assert stream.warmup_period == 0
    for lhs, rhs in zip(batch, streamed):
        np.testing.assert_allclose(lhs, rhs, rtol=1e-12, atol=1e-12, equal_nan=True)


def test_absolute_strength_index_oscillator_batch_single_param_matches_single():
    data = np.array([10.0, 10.0, 9.0, 9.0, 10.0, 11.0], dtype=np.float64)
    batch = mp.absolute_strength_index_oscillator_batch(
        data,
        ema_length_range=(3, 3, 0),
        signal_length_range=(4, 4, 0),
    )
    single = mp.absolute_strength_index_oscillator(data, ema_length=3, signal_length=4)

    assert batch["rows"] == 1
    assert batch["cols"] == len(data)
    np.testing.assert_array_equal(batch["ema_lengths"], np.array([3], dtype=np.uint64))
    np.testing.assert_array_equal(batch["signal_lengths"], np.array([4], dtype=np.uint64))
    np.testing.assert_allclose(
        batch["oscillator"][0], single[0], rtol=1e-12, atol=1e-12, equal_nan=True
    )
    np.testing.assert_allclose(
        batch["signal"][0], single[1], rtol=1e-12, atol=1e-12, equal_nan=True
    )
    np.testing.assert_allclose(
        batch["histogram"][0], single[2], rtol=1e-12, atol=1e-12, equal_nan=True
    )


def test_absolute_strength_index_oscillator_batch_metadata():
    data = np.linspace(95.0, 105.0, 64, dtype=np.float64)
    batch = mp.absolute_strength_index_oscillator_batch(
        data,
        ema_length_range=(3, 5, 2),
        signal_length_range=(4, 6, 2),
    )

    assert batch["rows"] == 4
    assert batch["cols"] == len(data)
    assert batch["oscillator"].shape == (4, len(data))
    np.testing.assert_array_equal(batch["ema_lengths"], np.array([3, 3, 5, 5], dtype=np.uint64))
    np.testing.assert_array_equal(
        batch["signal_lengths"], np.array([4, 6, 4, 6], dtype=np.uint64)
    )
