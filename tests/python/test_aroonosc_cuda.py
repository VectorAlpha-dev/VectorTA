"""Python CUDA binding tests for Aroon Oscillator (aroonosc)."""
import pytest
import numpy as np

try:
    import cupy as cp
except Exception:
    cp = None

try:
    import my_project as ti
except Exception:
    pytest.skip(
        "Python module not built; run `maturin develop --features python,cuda`",
        allow_module_level=True,
    )


def _has(attr: str) -> bool:
    return hasattr(ti, attr)


from test_utils import load_test_data


@pytest.mark.skipif(
    not _has("aroonosc_cuda_batch_dev") or cp is None,
    reason="CUDA bindings or CuPy not present",
)
def test_aroonosc_cuda_batch_dev_matches_cpu():
    td = load_test_data()
    high = td["high"].astype(np.float32)
    low = td["low"].astype(np.float32)

    handle = ti.aroonosc_cuda_batch_dev(high, low, (10, 30, 5), device_id=0)
    gpu = cp.asnumpy(cp.asarray(handle))


    rows = (30 - 10) // 5 + 1
    expect = []
    for L in range(10, 31, 5):
        expect.append(ti.aroonosc(high.astype(np.float64), low.astype(np.float64), L))
    expect = np.vstack(expect).astype(np.float32)


    assert gpu.shape == expect.shape
    tol = 8e-4
    assert np.allclose(np.nan_to_num(gpu, nan=0.0), np.nan_to_num(expect, nan=0.0), rtol=tol, atol=tol)


@pytest.mark.skipif(
    not _has("aroonosc_cuda_many_series_one_param_dev"),
    reason="CUDA bindings not present",
)
def test_aroonosc_cuda_many_series_one_param_shapes():
    rows, cols = 16, 256
    rng = np.random.default_rng(0)
    base = rng.normal(0, 1, size=(rows, cols)).astype(np.float32)
    high = base + 0.6
    low = base - 0.5
    handle = ti.aroonosc_cuda_many_series_one_param_dev(high, low, 14, device_id=0)
    assert hasattr(handle, "__cuda_array_interface__")

