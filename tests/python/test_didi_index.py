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


def test_didi_index_output_contract_and_manual_values():
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float64)
    short, long, crossover, crossunder = mp.didi_index(
        data, short_length=2, medium_length=3, long_length=4
    )

    assert len(short) == len(data)
    assert np.all(np.isnan(short[:3]))
    assert np.all(np.isnan(long[:3]))
    np.testing.assert_allclose(short[3], 3.5 / 3.0, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(long[3], 2.5 / 3.0, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(short[4], 4.5 / 4.0, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(long[4], 3.5 / 4.0, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(crossover[3], 0.0, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(crossunder[3], 0.0, rtol=0.0, atol=0.0)


def test_didi_index_kernel_parity():
    data = np.linspace(1.0, 20.0, 128, dtype=np.float64)
    auto = mp.didi_index(data, short_length=3, medium_length=8, long_length=20)
    scalar = mp.didi_index(
        data, short_length=3, medium_length=8, long_length=20, kernel="scalar"
    )

    for lhs, rhs in zip(auto, scalar):
        np.testing.assert_allclose(lhs, rhs, rtol=1e-12, atol=1e-12, equal_nan=True)


def test_didi_index_rejects_invalid_lengths():
    data = np.array([1.0, 2.0, 3.0], dtype=np.float64)

    with pytest.raises(ValueError, match="Invalid short_length"):
        mp.didi_index(data, short_length=0, medium_length=2, long_length=3)

    with pytest.raises(ValueError, match="Invalid medium_length"):
        mp.didi_index(data, short_length=1, medium_length=0, long_length=3)

    with pytest.raises(ValueError, match="Invalid long_length"):
        mp.didi_index(data, short_length=1, medium_length=2, long_length=0)


def test_didi_index_stream_matches_batch_with_reset():
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, np.nan, 3.0, 4.0, 5.0, 6.0], dtype=np.float64)
    batch = mp.didi_index(data, short_length=2, medium_length=3, long_length=4)
    stream = mp.DidiIndexStream(2, 3, 4)

    streamed = [[], [], [], []]
    for value in data:
        out = stream.update(float(value))
        if out is None:
            for series in streamed:
                series.append(np.nan)
        else:
            for series, component in zip(streamed, out):
                series.append(component)

    assert stream.warmup_period == 3
    for lhs, rhs in zip(batch, streamed):
        np.testing.assert_allclose(lhs, rhs, rtol=1e-12, atol=1e-12, equal_nan=True)

    assert np.isnan(batch[0][5])
    assert np.isnan(batch[0][8])
    assert np.isfinite(batch[0][9])


def test_didi_index_batch_single_param_matches_single():
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0, 1.0], dtype=np.float64)
    batch = mp.didi_index_batch(
        data,
        short_length_range=(2, 2, 0),
        medium_length_range=(3, 3, 0),
        long_length_range=(4, 4, 0),
    )
    single = mp.didi_index(data, short_length=2, medium_length=3, long_length=4)

    assert batch["rows"] == 1
    assert batch["cols"] == len(data)
    np.testing.assert_array_equal(batch["short_lengths"], np.array([2], dtype=np.uint64))
    np.testing.assert_array_equal(batch["medium_lengths"], np.array([3], dtype=np.uint64))
    np.testing.assert_array_equal(batch["long_lengths"], np.array([4], dtype=np.uint64))

    np.testing.assert_allclose(batch["short"][0], single[0], rtol=1e-12, atol=1e-12, equal_nan=True)
    np.testing.assert_allclose(batch["long"][0], single[1], rtol=1e-12, atol=1e-12, equal_nan=True)
    np.testing.assert_allclose(
        batch["crossover"][0], single[2], rtol=1e-12, atol=1e-12, equal_nan=True
    )
    np.testing.assert_allclose(
        batch["crossunder"][0], single[3], rtol=1e-12, atol=1e-12, equal_nan=True
    )


def test_didi_index_batch_metadata():
    data = np.linspace(1.0, 16.0, 32, dtype=np.float64)
    batch = mp.didi_index_batch(
        data,
        short_length_range=(2, 4, 2),
        medium_length_range=(3, 3, 0),
        long_length_range=(4, 6, 2),
    )

    assert batch["rows"] == 4
    assert batch["cols"] == len(data)
    assert batch["short"].shape == (4, len(data))
    np.testing.assert_array_equal(batch["short_lengths"], np.array([2, 2, 4, 4], dtype=np.uint64))
    np.testing.assert_array_equal(batch["medium_lengths"], np.array([3, 3, 3, 3], dtype=np.uint64))
    np.testing.assert_array_equal(batch["long_lengths"], np.array([4, 6, 4, 6], dtype=np.uint64))
