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
        data[i] = 100.0 + x * 0.05 + np.sin(x * 0.17) * 2.4 + np.cos(x * 0.04) * 0.8
    return data


def test_monotonicity_index_output_contract():
    data = sample_data(256)
    index, cumulative_mean, upper_bound = mp.monotonicity_index(
        data, length=20, mode="efficiency", index_smooth=5
    )

    assert len(index) == len(data)
    assert len(cumulative_mean) == len(data)
    assert len(upper_bound) == len(data)
    assert int(np.flatnonzero(~np.isnan(index))[0]) == 23
    assert int(np.flatnonzero(~np.isnan(cumulative_mean))[0]) == 23
    assert int(np.flatnonzero(~np.isnan(upper_bound))[0]) == 23
    assert np.isfinite(index[-1])
    assert np.isfinite(cumulative_mean[-1])
    assert np.isfinite(upper_bound[-1])


def test_monotonicity_index_rejects_invalid_parameters():
    data = sample_data(64)

    with pytest.raises(ValueError, match="Invalid length"):
        mp.monotonicity_index(data, length=1)

    with pytest.raises(ValueError, match="Invalid index_smooth"):
        mp.monotonicity_index(data, index_smooth=0)

    with pytest.raises(ValueError, match="Invalid mode"):
        mp.monotonicity_index(data, mode="bad")


def test_monotonicity_index_stream_matches_batch_with_reset():
    data = sample_data(192)
    data[96] = np.nan

    batch_index, batch_mean, batch_upper = mp.monotonicity_index(
        data, length=20, mode="efficiency", index_smooth=5
    )
    stream = mp.MonotonicityIndexStream(length=20, mode="efficiency", index_smooth=5)

    index = []
    cumulative_mean = []
    upper_bound = []
    for value in data:
        out = stream.update(float(value))
        if out is None:
            index.append(np.nan)
            cumulative_mean.append(np.nan)
            upper_bound.append(np.nan)
        else:
            idx, mean, upper = out
            index.append(idx)
            cumulative_mean.append(mean)
            upper_bound.append(upper)

    np.testing.assert_allclose(
        np.asarray(index), batch_index, rtol=1e-12, atol=1e-12, equal_nan=True
    )
    np.testing.assert_allclose(
        np.asarray(cumulative_mean),
        batch_mean,
        rtol=1e-12,
        atol=1e-12,
        equal_nan=True,
    )
    np.testing.assert_allclose(
        np.asarray(upper_bound),
        batch_upper,
        rtol=1e-12,
        atol=1e-12,
        equal_nan=True,
    )


def test_monotonicity_index_batch_single_param_matches_single():
    data = sample_data(192)
    batch = mp.monotonicity_index_batch(
        data,
        length_range=(20, 20, 0),
        index_smooth_range=(5, 5, 0),
        mode="efficiency",
    )
    index, cumulative_mean, upper_bound = mp.monotonicity_index(
        data, length=20, mode="efficiency", index_smooth=5
    )

    assert batch["rows"] == 1
    assert batch["cols"] == len(data)
    np.testing.assert_allclose(
        batch["index"][0], index, rtol=1e-12, atol=1e-12, equal_nan=True
    )
    np.testing.assert_allclose(
        batch["cumulative_mean"][0],
        cumulative_mean,
        rtol=1e-12,
        atol=1e-12,
        equal_nan=True,
    )
    np.testing.assert_allclose(
        batch["upper_bound"][0],
        upper_bound,
        rtol=1e-12,
        atol=1e-12,
        equal_nan=True,
    )


def test_monotonicity_index_batch_metadata():
    data = sample_data(160)
    batch = mp.monotonicity_index_batch(
        data,
        length_range=(18, 20, 2),
        index_smooth_range=(4, 5, 1),
        mode="complexity",
    )

    assert batch["rows"] == 4
    assert batch["cols"] == len(data)
    assert batch["index"].shape == (4, len(data))
    assert batch["cumulative_mean"].shape == (4, len(data))
    assert batch["upper_bound"].shape == (4, len(data))
    assert list(batch["modes"]) == ["complexity"] * 4
