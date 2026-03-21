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


def sample_close(length: int):
    out = np.full(length, np.nan, dtype=np.float64)
    prev = 100.0
    for i in range(2, length):
        x = float(i)
        value = prev + np.sin(x * 0.07) * 1.3 + np.cos(x * 0.03) * 0.6 + x * 0.02
        out[i] = value
        prev = value
    return out


def test_stochastic_distance_output_contract():
    close = sample_close(512)
    oscillator, signal = mp.stochastic_distance(
        close, lookback_length=80, length1=12, length2=3, ob_level=40, os_level=-40
    )

    assert len(oscillator) == len(close)
    assert len(signal) == len(close)
    first_valid = int(np.flatnonzero(~np.isnan(oscillator))[0])
    assert first_valid >= 91
    tail = signal[first_valid + 16 :]
    assert np.all(np.isnan(tail) | np.isin(tail, [-1.0, 0.0, 1.0]))


def test_stochastic_distance_rejects_invalid_parameters():
    close = sample_close(64)

    with pytest.raises(ValueError, match="Invalid lookback_length"):
        mp.stochastic_distance(close, lookback_length=0, length1=12, length2=3, ob_level=40, os_level=-40)

    with pytest.raises(ValueError, match="Invalid os_level"):
        mp.stochastic_distance(close, lookback_length=20, length1=12, length2=3, ob_level=40, os_level=10)


def test_stochastic_distance_stream_matches_batch_with_reset():
    close = sample_close(256)
    close[120] = np.nan

    batch_osc, batch_signal = mp.stochastic_distance(
        close, lookback_length=60, length1=10, length2=4, ob_level=35, os_level=-35
    )
    stream = mp.StochasticDistanceStream(60, 10, 4, 35, -35)

    osc = []
    signal = []
    for value in close:
        out = stream.update(float(value))
        if out is None:
            osc.append(np.nan)
            signal.append(np.nan)
        else:
            o, s = out
            osc.append(o)
            signal.append(s)

    np.testing.assert_allclose(np.asarray(osc), batch_osc, rtol=1e-12, atol=1e-12, equal_nan=True)
    np.testing.assert_allclose(
        np.asarray(signal), batch_signal, rtol=1e-12, atol=1e-12, equal_nan=True
    )


def test_stochastic_distance_batch_single_param_matches_single():
    close = sample_close(192)
    batch = mp.stochastic_distance_batch(
        close,
        lookback_length_range=(50, 50, 0),
        length1_range=(8, 8, 0),
        length2_range=(4, 4, 0),
        ob_level_range=(40, 40, 0),
        os_level_range=(-40, -40, 0),
    )
    oscillator, signal = mp.stochastic_distance(
        close, lookback_length=50, length1=8, length2=4, ob_level=40, os_level=-40
    )

    assert batch["rows"] == 1
    assert batch["cols"] == len(close)
    np.testing.assert_allclose(batch["oscillator"][0], oscillator, rtol=1e-12, atol=1e-12, equal_nan=True)
    np.testing.assert_allclose(batch["signal"][0], signal, rtol=1e-12, atol=1e-12, equal_nan=True)


def test_stochastic_distance_batch_metadata():
    close = sample_close(160)
    batch = mp.stochastic_distance_batch(
        close,
        lookback_length_range=(40, 60, 20),
        length1_range=(8, 8, 0),
        length2_range=(3, 5, 2),
        ob_level_range=(30, 40, 10),
        os_level_range=(-40, -30, 10),
    )

    assert batch["rows"] == 16
    assert batch["cols"] == len(close)
    assert batch["oscillator"].shape == (16, len(close))
    assert batch["signal"].shape == (16, len(close))
