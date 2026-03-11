import numpy as np
import pytest

try:
    import my_project as mp
except ImportError:
    try:
        import vector_ta as mp
    except ImportError:
        pytest.skip(
            "Python module not built. Run 'maturin develop --features python' first",
            allow_module_level=True,
        )


def sample_data(length: int, slots_per_day: int):
    data = np.empty(length, dtype=np.float64)
    for i in range(length):
        slot = float(i % slots_per_day)
        day = float(i // slots_per_day)
        data[i] = (
            1000.0
            + day * 5.0
            + np.sin(slot * 0.11) * 30.0
            + np.cos(slot * 0.03) * 12.0
            + (slot / slots_per_day) * 25.0
        )
    return data


def test_half_causal_estimator_output_contract():
    slots_per_day = 60
    data = sample_data(slots_per_day * 4, slots_per_day)
    out = mp.half_causal_estimator(data, slots_per_day, enable_expected_value=True)

    assert set(out.keys()) == {"estimate", "expected_value"}
    assert len(out["estimate"]) == len(data)
    assert len(out["expected_value"]) == len(data)
    assert np.isfinite(out["estimate"][-1])
    assert np.isfinite(out["expected_value"][-1])


def test_half_causal_estimator_rejects_invalid_parameters():
    data = sample_data(128, 32)

    with pytest.raises(ValueError, match="Invalid slots_per_day"):
        mp.half_causal_estimator(data, 1)

    with pytest.raises(ValueError, match="Invalid kernel_type"):
        mp.half_causal_estimator(data, 32, kernel_type="bad")


def test_half_causal_estimator_stream_matches_batch():
    slots_per_day = 48
    data = sample_data(slots_per_day * 5, slots_per_day)
    batch = mp.half_causal_estimator(
        data,
        slots_per_day,
        enable_expected_value=True,
        extra_smoothing=2,
    )
    stream = mp.HalfCausalEstimatorStream(
        slots_per_day,
        enable_expected_value=True,
        extra_smoothing=2,
    )
    est = []
    exp = []
    for value in data:
        estimate, expected_value = stream.update(float(value))
        est.append(np.nan if estimate is None else estimate)
        exp.append(np.nan if expected_value is None else expected_value)

    np.testing.assert_allclose(
        np.asarray(est), batch["estimate"], rtol=1e-12, atol=1e-12, equal_nan=True
    )
    np.testing.assert_allclose(
        np.asarray(exp), batch["expected_value"], rtol=1e-12, atol=1e-12, equal_nan=True
    )


def test_half_causal_estimator_batch_single_matches_single():
    slots_per_day = 60
    data = sample_data(slots_per_day * 4, slots_per_day)
    batch = mp.half_causal_estimator_batch(
        data,
        slots_per_day,
        enable_expected_value=True,
    )
    single = mp.half_causal_estimator(
        data,
        slots_per_day,
        enable_expected_value=True,
    )

    assert batch["rows"] == 1
    assert batch["cols"] == len(data)
    np.testing.assert_allclose(
        batch["estimate"][0], single["estimate"], rtol=1e-12, atol=1e-12, equal_nan=True
    )
    np.testing.assert_allclose(
        batch["expected_value"][0],
        single["expected_value"],
        rtol=1e-12,
        atol=1e-12,
        equal_nan=True,
    )


def test_half_causal_estimator_batch_metadata():
    slots_per_day = 48
    data = sample_data(slots_per_day * 4, slots_per_day)
    batch = mp.half_causal_estimator_batch(
        data,
        slots_per_day,
        data_period_range=(4, 5, 1),
        extra_smoothing_range=(0, 1, 1),
        enable_expected_value=True,
    )

    assert batch["rows"] == 4
    assert batch["cols"] == len(data)
    np.testing.assert_array_equal(batch["data_periods"], np.array([4, 4, 5, 5], dtype=np.uint64))
    np.testing.assert_array_equal(
        batch["extra_smoothings"], np.array([0, 1, 0, 1], dtype=np.uint64)
    )
