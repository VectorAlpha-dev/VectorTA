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


def sample_data(length: int):
    data = np.empty(length, dtype=np.float64)
    for i in range(length):
        x = float(i)
        data[i] = 100.0 + x * 0.03 + np.sin(x * 0.09) * 2.1 + np.cos(x * 0.02) * 0.7
    return data


def test_normalized_resonator_output_contract():
    data = sample_data(256)
    oscillator, signal = mp.normalized_resonator(
        data, period=48, delta=0.4, lookback_mult=1.2, signal_length=7
    )

    assert len(oscillator) == len(data)
    assert len(signal) == len(data)
    assert int(np.flatnonzero(~np.isnan(oscillator))[0]) == 2
    assert int(np.flatnonzero(~np.isnan(signal))[0]) == 2
    assert np.isfinite(oscillator[-1])
    assert np.isfinite(signal[-1])


def test_normalized_resonator_rejects_invalid_parameters():
    data = sample_data(64)

    with pytest.raises(ValueError, match="Invalid period"):
        mp.normalized_resonator(data, period=1)

    with pytest.raises(ValueError, match="Invalid delta"):
        mp.normalized_resonator(data, delta=0.0)

    with pytest.raises(ValueError, match="Invalid signal_length"):
        mp.normalized_resonator(data, signal_length=0)


def test_normalized_resonator_stream_matches_batch_with_reset():
    data = sample_data(192)
    data[96] = np.nan

    batch_oscillator, batch_signal = mp.normalized_resonator(
        data, period=48, delta=0.4, lookback_mult=1.2, signal_length=7
    )
    stream = mp.NormalizedResonatorStream(
        period=48, delta=0.4, lookback_mult=1.2, signal_length=7
    )

    oscillator = []
    signal = []
    for value in data:
        out = stream.update(float(value))
        if out is None:
            oscillator.append(np.nan)
            signal.append(np.nan)
        else:
            osc, sig = out
            oscillator.append(osc)
            signal.append(sig)

    np.testing.assert_allclose(
        np.asarray(oscillator), batch_oscillator, rtol=1e-12, atol=1e-12, equal_nan=True
    )
    np.testing.assert_allclose(
        np.asarray(signal), batch_signal, rtol=1e-12, atol=1e-12, equal_nan=True
    )


def test_normalized_resonator_batch_single_param_matches_single():
    data = sample_data(192)
    batch = mp.normalized_resonator_batch(
        data,
        period_range=(48, 48, 0),
        delta_range=(0.4, 0.4, 0.0),
        lookback_mult_range=(1.2, 1.2, 0.0),
        signal_length_range=(7, 7, 0),
    )
    oscillator, signal = mp.normalized_resonator(
        data, period=48, delta=0.4, lookback_mult=1.2, signal_length=7
    )

    assert batch["rows"] == 1
    assert batch["cols"] == len(data)
    np.testing.assert_allclose(
        batch["oscillator"][0], oscillator, rtol=1e-12, atol=1e-12, equal_nan=True
    )
    np.testing.assert_allclose(
        batch["signal"][0], signal, rtol=1e-12, atol=1e-12, equal_nan=True
    )


def test_normalized_resonator_batch_metadata():
    data = sample_data(160)
    batch = mp.normalized_resonator_batch(
        data,
        period_range=(40, 44, 4),
        delta_range=(0.3, 0.5, 0.2),
        lookback_mult_range=(1.0, 1.2, 0.2),
        signal_length_range=(5, 6, 1),
    )

    assert batch["rows"] == 16
    assert batch["cols"] == len(data)
    assert batch["oscillator"].shape == (16, len(data))
    assert batch["signal"].shape == (16, len(data))
