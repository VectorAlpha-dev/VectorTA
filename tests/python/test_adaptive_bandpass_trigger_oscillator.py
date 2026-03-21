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


def test_adaptive_bandpass_trigger_oscillator_output_contract():
    close = sample_close(512)
    in_phase, lead = mp.adaptive_bandpass_trigger_oscillator(close, delta=0.1, alpha=0.07)

    assert len(in_phase) == len(close)
    assert len(lead) == len(close)
    assert int(np.flatnonzero(~np.isnan(in_phase))[0]) >= 13
    assert int(np.flatnonzero(~np.isnan(lead))[0]) >= 14
    assert np.isfinite(in_phase[-1])
    assert np.isfinite(lead[-1])


def test_adaptive_bandpass_trigger_oscillator_rejects_invalid_parameters():
    close = sample_close(64)

    with pytest.raises(ValueError, match="Invalid delta"):
        mp.adaptive_bandpass_trigger_oscillator(close, delta=0.0, alpha=0.07)

    with pytest.raises(ValueError, match="Invalid alpha"):
        mp.adaptive_bandpass_trigger_oscillator(close, delta=0.1, alpha=1.0)


def test_adaptive_bandpass_trigger_oscillator_stream_matches_batch_with_reset():
    close = sample_close(256)
    close[120] = np.nan

    batch_in_phase, batch_lead = mp.adaptive_bandpass_trigger_oscillator(
        close, delta=0.12, alpha=0.08
    )
    stream = mp.AdaptiveBandpassTriggerOscillatorStream(0.12, 0.08)

    in_phase = []
    lead = []
    for value in close:
        out = stream.update(float(value))
        if out is None:
            in_phase.append(np.nan)
            lead.append(np.nan)
        else:
            bp, ld = out
            in_phase.append(bp)
            lead.append(ld)

    np.testing.assert_allclose(
        np.asarray(in_phase), batch_in_phase, rtol=1e-12, atol=1e-12, equal_nan=True
    )
    np.testing.assert_allclose(
        np.asarray(lead), batch_lead, rtol=1e-12, atol=1e-12, equal_nan=True
    )


def test_adaptive_bandpass_trigger_oscillator_batch_single_param_matches_single():
    close = sample_close(192)
    batch = mp.adaptive_bandpass_trigger_oscillator_batch(
        close,
        delta_range=(0.1, 0.1, 0.0),
        alpha_range=(0.07, 0.07, 0.0),
    )
    in_phase, lead = mp.adaptive_bandpass_trigger_oscillator(close, delta=0.1, alpha=0.07)

    assert batch["rows"] == 1
    assert batch["cols"] == len(close)
    np.testing.assert_allclose(batch["in_phase"][0], in_phase, rtol=1e-12, atol=1e-12, equal_nan=True)
    np.testing.assert_allclose(batch["lead"][0], lead, rtol=1e-12, atol=1e-12, equal_nan=True)


def test_adaptive_bandpass_trigger_oscillator_batch_metadata():
    close = sample_close(160)
    batch = mp.adaptive_bandpass_trigger_oscillator_batch(
        close,
        delta_range=(0.08, 0.12, 0.04),
        alpha_range=(0.05, 0.09, 0.02),
    )

    assert batch["rows"] == 6
    assert batch["cols"] == len(close)
    assert batch["in_phase"].shape == (6, len(close))
    assert batch["lead"].shape == (6, len(close))
