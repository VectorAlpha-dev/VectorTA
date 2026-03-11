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


def test_multi_length_stochastic_average_output_contract():
    data = np.linspace(90.0, 110.0, 256, dtype=np.float64)
    values = mp.multi_length_stochastic_average(data)

    assert len(values) == len(data)
    finite_idx = np.flatnonzero(np.isfinite(values))
    assert finite_idx.size > 0
    assert finite_idx[0] >= 22
    finite = values[np.isfinite(values)]
    assert np.all(finite >= -1e-9)
    assert np.all(finite <= 100.0 + 1e-9)


def test_multi_length_stochastic_average_kernel_parity():
    data = np.array(
        [100.0 + i * 0.04 + np.sin(i * 0.17) * 1.8 + np.cos(i * 0.03) * 0.4 for i in range(256)],
        dtype=np.float64,
    )
    auto = mp.multi_length_stochastic_average(
        data, length=12, presmooth=5, premethod="lsma", postsmooth=4, postmethod="sma"
    )
    scalar = mp.multi_length_stochastic_average(
        data,
        length=12,
        presmooth=5,
        premethod="lsma",
        postsmooth=4,
        postmethod="sma",
        kernel="scalar",
    )
    np.testing.assert_allclose(auto, scalar, rtol=1e-12, atol=1e-12, equal_nan=True)


def test_multi_length_stochastic_average_rejects_invalid_parameters():
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)

    with pytest.raises(ValueError, match="Invalid length"):
        mp.multi_length_stochastic_average(data, length=3)

    with pytest.raises(ValueError, match="Invalid premethod"):
        mp.multi_length_stochastic_average(data, premethod="ema")

    with pytest.raises(ValueError, match="Invalid postsmooth"):
        mp.multi_length_stochastic_average(data, postsmooth=0)


def test_multi_length_stochastic_average_stream_matches_batch_with_reset():
    data = np.array(
        [100.0 + i * 0.05 + np.sin(i * 0.15) * 1.4 for i in range(220)],
        dtype=np.float64,
    )
    data[110] = np.nan
    batch = mp.multi_length_stochastic_average(
        data, length=12, presmooth=5, premethod="lsma", postsmooth=4, postmethod="sma"
    )
    stream = mp.MultiLengthStochasticAverageStream(12, 5, "lsma", 4, "sma")

    streamed = []
    for value in data:
        out = stream.update(float(value))
        streamed.append(np.nan if out is None else out)

    assert stream.warmup_period == 18
    np.testing.assert_allclose(batch, streamed, rtol=1e-12, atol=1e-12, equal_nan=True)


def test_multi_length_stochastic_average_batch_single_param_matches_single():
    data = np.array(
        [100.0 + i * 0.04 + np.sin(i * 0.17) * 1.8 + np.cos(i * 0.03) * 0.4 for i in range(128)],
        dtype=np.float64,
    )
    batch = mp.multi_length_stochastic_average_batch(
        data,
        length_range=(12, 12, 0),
        presmooth_range=(5, 5, 0),
        premethod="lsma",
        postsmooth_range=(4, 4, 0),
        postmethod="sma",
    )
    single = mp.multi_length_stochastic_average(
        data, length=12, presmooth=5, premethod="lsma", postsmooth=4, postmethod="sma"
    )

    assert batch["rows"] == 1
    assert batch["cols"] == len(data)
    np.testing.assert_array_equal(batch["lengths"], np.array([12], dtype=np.uint64))
    np.testing.assert_array_equal(batch["presmooths"], np.array([5], dtype=np.uint64))
    np.testing.assert_array_equal(batch["postsmooths"], np.array([4], dtype=np.uint64))
    assert list(batch["premethods"]) == ["lsma"]
    assert list(batch["postmethods"]) == ["sma"]
    np.testing.assert_allclose(
        batch["values"][0], single, rtol=1e-12, atol=1e-12, equal_nan=True
    )


def test_multi_length_stochastic_average_batch_metadata():
    data = np.array(
        [100.0 + i * 0.04 + np.sin(i * 0.17) * 1.8 + np.cos(i * 0.03) * 0.4 for i in range(96)],
        dtype=np.float64,
    )
    batch = mp.multi_length_stochastic_average_batch(
        data,
        length_range=(10, 14, 2),
        presmooth_range=(4, 6, 2),
        premethod="sma",
        postsmooth_range=(3, 5, 2),
        postmethod="lsma",
    )

    assert batch["rows"] == 12
    assert batch["cols"] == len(data)
    assert batch["values"].shape == (12, len(data))
    np.testing.assert_array_equal(
        batch["lengths"],
        np.array([10, 10, 10, 10, 12, 12, 12, 12, 14, 14, 14, 14], dtype=np.uint64),
    )
    np.testing.assert_array_equal(
        batch["presmooths"],
        np.array([4, 4, 6, 6, 4, 4, 6, 6, 4, 4, 6, 6], dtype=np.uint64),
    )
    np.testing.assert_array_equal(
        batch["postsmooths"],
        np.array([3, 5, 3, 5, 3, 5, 3, 5, 3, 5, 3, 5], dtype=np.uint64),
    )
    assert list(batch["premethods"]) == ["sma"] * 12
    assert list(batch["postmethods"]) == ["lsma"] * 12
