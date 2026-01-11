"""Python CUDA binding tests for VOSS indicator."""
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

from test_utils import load_test_data


def _has(attr: str) -> bool:
    return hasattr(ti, attr)


@pytest.mark.skipif(
    not _has("voss_cuda_batch_dev") or cp is None,
    reason="CUDA bindings or CuPy not present",
)
def test_voss_cuda_batch_dev_matches_cpu():
    td = load_test_data()
    close = td["close"].astype(np.float32)

    
    out = ti.voss_cuda_batch_dev(close, (10, 18, 4), (1, 3, 1), (0.15, 0.35, 0.10), device_id=0)
    voss_dev = out["voss"]
    filt_dev = out["filt"]

    voss = cp.asnumpy(cp.asarray(voss_dev)).astype(np.float32)
    filt = cp.asnumpy(cp.asarray(filt_dev)).astype(np.float32)

    
    rows = voss.shape[0]
    expect_v = []
    expect_f = []
    periods = list(range(10, 19, 4))
    predicts = [1, 2, 3]
    bws = [0.15, 0.25, 0.35]
    combos = [(p, q, b) for p in periods for q in predicts for b in bws]
    close_f64 = close.astype(np.float64)
    for p, q, b in combos:
        v, f = ti.voss(close_f64, p, q, b)
        expect_v.append(v.astype(np.float32))
        expect_f.append(f.astype(np.float32))
    expect_v = np.vstack(expect_v)
    expect_f = np.vstack(expect_f)

    assert voss.shape == expect_v.shape
    assert filt.shape == expect_f.shape
    tol = 1e-2
    np.testing.assert_allclose(np.nan_to_num(voss, nan=0.0), np.nan_to_num(expect_v, nan=0.0), rtol=tol, atol=tol)
    np.testing.assert_allclose(np.nan_to_num(filt, nan=0.0), np.nan_to_num(expect_f, nan=0.0), rtol=tol, atol=tol)


@pytest.mark.skipif(
    not _has("voss_cuda_many_series_one_param_dev") or cp is None,
    reason="CUDA bindings or CuPy not present",
)
def test_voss_cuda_many_series_one_param_shapes_and_basic_check():
    rows, cols = 2048, 8
    base = np.linspace(0, 1, rows, dtype=np.float32)
    data_tm = np.stack([np.sin(base * (i + 1)) + 0.001 * i for i in range(cols)], axis=1).astype(np.float32)
    handle_v, handle_f = ti.voss_cuda_many_series_one_param_dev(data_tm, 20, 3, 0.25, device_id=0)
    v_tm = cp.asnumpy(cp.asarray(handle_v))
    f_tm = cp.asnumpy(cp.asarray(handle_f))
    assert v_tm.shape == (rows, cols)
    assert f_tm.shape == (rows, cols)

