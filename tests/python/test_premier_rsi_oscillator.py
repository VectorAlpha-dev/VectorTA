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


def test_premier_rsi_oscillator_output_contract_on_constant_data():
    data = np.full(8, 10.0, dtype=np.float64)
    values = mp.premier_rsi_oscillator(data, rsi_length=2, stoch_length=2, smooth_length=4)

    assert len(values) == len(data)
    assert np.all(np.isnan(values[:3]))
    np.testing.assert_allclose(values[3:], 0.0, rtol=0.0, atol=1e-12)


def test_premier_rsi_oscillator_kernel_parity():
    data = np.linspace(90.0, 110.0, 256, dtype=np.float64)
    auto = mp.premier_rsi_oscillator(
        data, rsi_length=14, stoch_length=8, smooth_length=25
    )
    scalar = mp.premier_rsi_oscillator(
        data,
        rsi_length=14,
        stoch_length=8,
        smooth_length=25,
        kernel="scalar",
    )
    np.testing.assert_allclose(auto, scalar, rtol=1e-12, atol=1e-12, equal_nan=True)


def test_premier_rsi_oscillator_rejects_invalid_parameters():
    data = np.array([1.0, 2.0, 3.0], dtype=np.float64)

    with pytest.raises(ValueError, match="Invalid rsi_length"):
        mp.premier_rsi_oscillator(data, rsi_length=0, stoch_length=2, smooth_length=4)

    with pytest.raises(ValueError, match="Invalid stoch_length"):
        mp.premier_rsi_oscillator(data, rsi_length=2, stoch_length=0, smooth_length=4)

    with pytest.raises(ValueError, match="Invalid smooth_length"):
        mp.premier_rsi_oscillator(data, rsi_length=2, stoch_length=2, smooth_length=0)


def test_premier_rsi_oscillator_stream_matches_batch_with_reset():
    data = np.array(
        [10.0, 11.0, 10.5, 10.0, 10.2, np.nan, 10.0, 10.0, 10.1, 10.2],
        dtype=np.float64,
    )
    batch = mp.premier_rsi_oscillator(data, rsi_length=2, stoch_length=2, smooth_length=4)
    stream = mp.PremierRsiOscillatorStream(2, 2, 4)

    streamed = []
    for value in data:
        out = stream.update(float(value))
        streamed.append(np.nan if out is None else out)

    assert stream.warmup_period == 3
    np.testing.assert_allclose(batch, streamed, rtol=1e-12, atol=1e-12, equal_nan=True)


def test_premier_rsi_oscillator_batch_single_param_matches_single():
    data = np.linspace(95.0, 105.0, 64, dtype=np.float64)
    batch = mp.premier_rsi_oscillator_batch(
        data,
        rsi_length_range=(14, 14, 0),
        stoch_length_range=(8, 8, 0),
        smooth_length_range=(25, 25, 0),
    )
    single = mp.premier_rsi_oscillator(
        data, rsi_length=14, stoch_length=8, smooth_length=25
    )

    assert batch["rows"] == 1
    assert batch["cols"] == len(data)
    np.testing.assert_array_equal(batch["rsi_lengths"], np.array([14], dtype=np.uint64))
    np.testing.assert_array_equal(batch["stoch_lengths"], np.array([8], dtype=np.uint64))
    np.testing.assert_array_equal(batch["smooth_lengths"], np.array([25], dtype=np.uint64))
    np.testing.assert_allclose(
        batch["values"][0], single, rtol=1e-12, atol=1e-12, equal_nan=True
    )


def test_premier_rsi_oscillator_batch_metadata():
    data = np.linspace(95.0, 105.0, 64, dtype=np.float64)
    batch = mp.premier_rsi_oscillator_batch(
        data,
        rsi_length_range=(14, 16, 2),
        stoch_length_range=(8, 10, 2),
        smooth_length_range=(25, 27, 2),
    )

    assert batch["rows"] == 8
    assert batch["cols"] == len(data)
    assert batch["values"].shape == (8, len(data))
    np.testing.assert_array_equal(
        batch["rsi_lengths"], np.array([14, 14, 14, 14, 16, 16, 16, 16], dtype=np.uint64)
    )
    np.testing.assert_array_equal(
        batch["stoch_lengths"], np.array([8, 8, 10, 10, 8, 8, 10, 10], dtype=np.uint64)
    )
    np.testing.assert_array_equal(
        batch["smooth_lengths"], np.array([25, 27, 25, 27, 25, 27, 25, 27], dtype=np.uint64)
    )
