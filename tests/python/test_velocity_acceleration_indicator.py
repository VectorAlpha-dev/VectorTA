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
        data[i] = 100.0 + x * 0.05 + np.sin(x * 0.13) * 2.5 + np.cos(x * 0.04) * 0.8
    return data


def test_velocity_acceleration_indicator_output_contract():
    data = sample_data(256)
    values = mp.velocity_acceleration_indicator(data, length=21, smooth_length=5)

    assert len(values) == len(data)
    assert int(np.flatnonzero(~np.isnan(values))[0]) == 4
    assert np.isfinite(values[-1])


def test_velocity_acceleration_indicator_rejects_invalid_parameters():
    data = sample_data(32)

    with pytest.raises(ValueError, match="Invalid length"):
        mp.velocity_acceleration_indicator(data, length=1, smooth_length=5)

    with pytest.raises(ValueError, match="Invalid smooth_length"):
        mp.velocity_acceleration_indicator(data, length=21, smooth_length=0)


def test_velocity_acceleration_indicator_stream_matches_batch_with_reset():
    data = sample_data(180)
    data[90] = np.nan

    batch = mp.velocity_acceleration_indicator(data, length=21, smooth_length=5)
    stream = mp.VelocityAccelerationIndicatorStream(length=21, smooth_length=5)
    streamed = []
    for value in data:
        out = stream.update(float(value))
        streamed.append(np.nan if out is None else out)

    np.testing.assert_allclose(
        np.asarray(streamed), batch, rtol=1e-12, atol=1e-12, equal_nan=True
    )


def test_velocity_acceleration_indicator_batch_single_param_matches_single():
    data = sample_data(192)
    batch = mp.velocity_acceleration_indicator_batch(
        data,
        length_range=(21, 21, 0),
        smooth_length_range=(5, 5, 0),
    )
    values = mp.velocity_acceleration_indicator(data, length=21, smooth_length=5)

    assert batch["rows"] == 1
    assert batch["cols"] == len(data)
    np.testing.assert_allclose(
        batch["values"][0], values, rtol=1e-12, atol=1e-12, equal_nan=True
    )


def test_velocity_acceleration_indicator_batch_metadata():
    data = sample_data(160)
    batch = mp.velocity_acceleration_indicator_batch(
        data,
        length_range=(21, 23, 2),
        smooth_length_range=(4, 5, 1),
    )

    assert batch["rows"] == 4
    assert batch["cols"] == len(data)
    np.testing.assert_array_equal(batch["lengths"], np.array([21, 21, 23, 23], dtype=np.uint64))
    np.testing.assert_array_equal(
        batch["smooth_lengths"], np.array([4, 5, 4, 5], dtype=np.uint64)
    )
