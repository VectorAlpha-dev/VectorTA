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


def sample_data():
    return np.array(
        [
            100.0,
            100.5,
            101.0,
            100.7,
            100.2,
            99.8,
            99.5,
            99.7,
            100.0,
            100.4,
            100.8,
            101.0,
            100.9,
            100.7,
            100.4,
            100.1,
            99.9,
            np.nan,
            100.0,
            100.3,
            100.6,
            100.8,
            100.7,
            100.5,
            100.2,
            100.0,
            99.8,
            99.9,
            100.1,
            100.4,
            100.7,
            100.9,
            101.0,
            100.8,
            100.5,
        ],
        dtype=np.float64,
    )


def test_ehlers_linear_extrapolation_predictor_output_contract_on_constant_series():
    data = np.full(24, 100.0, dtype=np.float64)
    prediction, filt, state, go_long, go_short = mp.ehlers_linear_extrapolation_predictor(
        data,
        high_pass_length=8,
        low_pass_length=4,
        gain=0.7,
        bars_forward=3,
        signal_mode="predict_filter_crosses",
    )

    assert len(prediction) == len(data)
    assert np.all(np.isnan(prediction[:15]))
    assert np.all(np.isnan(filt[:15]))
    assert np.all(np.isnan(state[:15]))
    assert np.all(np.isnan(go_long[:15]))
    assert np.all(np.isnan(go_short[:15]))
    np.testing.assert_allclose(prediction[15:], 0.0, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(filt[15:], 0.0, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(state[15:], 0.0, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(go_long[15:], 0.0, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(go_short[15:], 0.0, rtol=1e-12, atol=1e-12)


def test_ehlers_linear_extrapolation_predictor_rejects_invalid_params():
    data = np.linspace(100.0, 110.0, 32, dtype=np.float64)

    with pytest.raises(ValueError, match="Invalid bars_forward"):
        mp.ehlers_linear_extrapolation_predictor(
            data,
            high_pass_length=8,
            low_pass_length=4,
            gain=0.7,
            bars_forward=11,
            signal_mode="predict_filter_crosses",
        )

    with pytest.raises(ValueError, match="Invalid signal_mode"):
        mp.ehlers_linear_extrapolation_predictor(
            data,
            high_pass_length=8,
            low_pass_length=4,
            gain=0.7,
            bars_forward=3,
            signal_mode="bad_mode",
        )


def test_ehlers_linear_extrapolation_predictor_stream_matches_batch_with_reset():
    data = sample_data()
    batch = mp.ehlers_linear_extrapolation_predictor(
        data,
        high_pass_length=8,
        low_pass_length=4,
        gain=0.7,
        bars_forward=3,
        signal_mode="predict_middle_crosses",
    )
    stream = mp.EhlersLinearExtrapolationPredictorStream(
        8, 4, 0.7, 3, "predict_middle_crosses"
    )

    streamed = [[] for _ in range(5)]
    for value in data:
        out = stream.update(float(value))
        if out is None:
            for series in streamed:
                series.append(np.nan)
        else:
            for series, component in zip(streamed, out):
                series.append(component)

    assert stream.warmup_period == 15
    for lhs, rhs in zip(batch, streamed):
        np.testing.assert_allclose(lhs, rhs, rtol=1e-12, atol=1e-12, equal_nan=True)

    assert np.isnan(batch[0][17])
    assert np.isnan(batch[0][32])
    assert np.isfinite(batch[0][-1])


def test_ehlers_linear_extrapolation_predictor_batch_single_param_matches_single():
    data = np.linspace(90.0, 110.0, 96, dtype=np.float64)
    batch = mp.ehlers_linear_extrapolation_predictor_batch(
        data,
        high_pass_length_range=(8, 8, 0),
        low_pass_length_range=(4, 4, 0),
        gain_range=(0.7, 0.7, 0.0),
        bars_forward_range=(3, 3, 0),
        signal_mode="filter_middle_crosses",
    )
    single = mp.ehlers_linear_extrapolation_predictor(
        data,
        high_pass_length=8,
        low_pass_length=4,
        gain=0.7,
        bars_forward=3,
        signal_mode="filter_middle_crosses",
    )

    assert batch["rows"] == 1
    assert batch["cols"] == len(data)
    np.testing.assert_array_equal(batch["high_pass_lengths"], np.array([8], dtype=np.uint64))
    np.testing.assert_array_equal(batch["low_pass_lengths"], np.array([4], dtype=np.uint64))
    np.testing.assert_array_equal(batch["gains"], np.array([0.7]))
    np.testing.assert_array_equal(batch["bars_forwards"], np.array([3], dtype=np.uint64))
    assert batch["signal_modes"] == ["filter_middle_crosses"]
    np.testing.assert_allclose(
        batch["prediction"][0], single[0], rtol=1e-12, atol=1e-12, equal_nan=True
    )
    np.testing.assert_allclose(
        batch["filter"][0], single[1], rtol=1e-12, atol=1e-12, equal_nan=True
    )
    np.testing.assert_allclose(
        batch["state"][0], single[2], rtol=1e-12, atol=1e-12, equal_nan=True
    )
    np.testing.assert_allclose(
        batch["go_long"][0], single[3], rtol=1e-12, atol=1e-12, equal_nan=True
    )
    np.testing.assert_allclose(
        batch["go_short"][0], single[4], rtol=1e-12, atol=1e-12, equal_nan=True
    )


def test_ehlers_linear_extrapolation_predictor_batch_metadata():
    data = np.linspace(95.0, 120.0, 96, dtype=np.float64)
    batch = mp.ehlers_linear_extrapolation_predictor_batch(
        data,
        high_pass_length_range=(8, 10, 2),
        low_pass_length_range=(4, 6, 2),
        gain_range=(0.7, 0.7, 0.0),
        bars_forward_range=(3, 5, 2),
        signal_mode="predict_filter_crosses",
    )

    assert batch["rows"] == 8
    assert batch["cols"] == len(data)
    assert batch["prediction"].shape == (8, len(data))
    np.testing.assert_array_equal(
        batch["high_pass_lengths"], np.array([8, 8, 8, 8, 10, 10, 10, 10], dtype=np.uint64)
    )
    np.testing.assert_array_equal(
        batch["low_pass_lengths"], np.array([4, 4, 6, 6, 4, 4, 6, 6], dtype=np.uint64)
    )
    np.testing.assert_array_equal(
        batch["gains"], np.array([0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7])
    )
    np.testing.assert_array_equal(
        batch["bars_forwards"], np.array([3, 5, 3, 5, 3, 5, 3, 5], dtype=np.uint64)
    )
    assert batch["signal_modes"] == ["predict_filter_crosses"] * 8
