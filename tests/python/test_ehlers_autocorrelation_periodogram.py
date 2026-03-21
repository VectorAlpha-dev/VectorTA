import math

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


def cycle_data(length=256, period=20.0):
    x = np.arange(length, dtype=np.float64)
    phase = 2.0 * math.pi * x / period
    return np.sin(phase) + 0.15 * np.cos(phase * 0.5)


def test_eacp_output_contract_on_cycle():
    data = cycle_data()
    dominant_cycle, normalized_power = mp.ehlers_autocorrelation_periodogram(data)

    assert len(dominant_cycle) == len(data)
    assert len(normalized_power) == len(data)
    assert np.all(np.isnan(dominant_cycle[:50]))
    assert np.isfinite(dominant_cycle[-1])
    assert np.isfinite(normalized_power[-1])
    assert 15.0 < dominant_cycle[-1] < 25.0
    assert 0.0 <= normalized_power[-1] <= 1.0


def test_eacp_rejects_invalid_params():
    data = cycle_data(64, 16.0)

    with pytest.raises(ValueError, match="Invalid min_period"):
        mp.ehlers_autocorrelation_periodogram(
            data, min_period=2, max_period=32, avg_length=3, enhance=True
        )

    with pytest.raises(ValueError, match="Invalid period order"):
        mp.ehlers_autocorrelation_periodogram(
            data, min_period=16, max_period=8, avg_length=3, enhance=True
        )


def test_eacp_stream_matches_batch_with_reset():
    data = cycle_data(192, 18.0)
    data[90] = np.nan

    batch = mp.ehlers_autocorrelation_periodogram(
        data, min_period=8, max_period=32, avg_length=3, enhance=True
    )
    stream = mp.EhlersAutocorrelationPeriodogramStream(8, 32, 3, True)

    streamed_dom = []
    streamed_pwr = []
    for value in data:
        out = stream.update(float(value))
        if out is None:
            streamed_dom.append(np.nan)
            streamed_pwr.append(np.nan)
        else:
            streamed_dom.append(out[0])
            streamed_pwr.append(out[1])

    assert stream.warmup_period == 34
    np.testing.assert_allclose(
        batch[0], np.array(streamed_dom), rtol=1e-9, atol=1e-9, equal_nan=True
    )
    np.testing.assert_allclose(
        batch[1], np.array(streamed_pwr), rtol=1e-9, atol=1e-9, equal_nan=True
    )
    assert np.isnan(batch[0][90])
    assert np.isnan(batch[0][120])
    assert np.isfinite(batch[0][-1])


def test_eacp_batch_single_param_matches_single():
    data = cycle_data(160, 22.0)
    batch = mp.ehlers_autocorrelation_periodogram_batch(
        data,
        min_period_range=(8, 8, 0),
        max_period_range=(48, 48, 0),
        avg_length_range=(3, 3, 0),
        enhance=True,
    )
    single = mp.ehlers_autocorrelation_periodogram(
        data, min_period=8, max_period=48, avg_length=3, enhance=True
    )

    assert batch["rows"] == 1
    assert batch["cols"] == len(data)
    np.testing.assert_allclose(
        batch["dominant_cycle"][0], single[0], rtol=1e-9, atol=1e-9, equal_nan=True
    )
    np.testing.assert_allclose(
        batch["normalized_power"][0], single[1], rtol=1e-9, atol=1e-9, equal_nan=True
    )


def test_eacp_batch_metadata():
    data = cycle_data(160, 21.0)
    batch = mp.ehlers_autocorrelation_periodogram_batch(
        data,
        min_period_range=(8, 10, 2),
        max_period_range=(24, 28, 4),
        avg_length_range=(3, 3, 0),
        enhance=False,
    )

    assert batch["rows"] == 4
    assert batch["cols"] == len(data)
    assert batch["dominant_cycle"].shape == (4, len(data))
    np.testing.assert_array_equal(batch["min_periods"], np.array([8, 8, 10, 10], dtype=np.uint64))
    np.testing.assert_array_equal(batch["max_periods"], np.array([24, 28, 24, 28], dtype=np.uint64))
    np.testing.assert_array_equal(batch["avg_lengths"], np.array([3, 3, 3, 3], dtype=np.uint64))
    assert batch["enhance_flags"] == [False, False, False, False]
