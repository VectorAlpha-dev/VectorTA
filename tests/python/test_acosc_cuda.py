"""Python CUDA binding tests for ACOSC."""
import pytest
import numpy as np

try:
    import my_project as ti
except Exception:
    pytest.skip("Python module not built; run maturin develop --features python", allow_module_level=True)


def _has(attr):
    return hasattr(ti, attr)


from test_utils import load_test_data


@pytest.mark.skipif(not _has("acosc_cuda_batch_dev"), reason="acosc CUDA bindings not present")
def test_acosc_cuda_batch_dev_matches_cpu():
    td = load_test_data()
    high = td["high"].astype(np.float32)
    low = td["low"].astype(np.float32)

    
    osc_dev, chg_dev = ti.acosc_cuda_batch_dev(high, low, device_id=0)

    
    try:
        import cupy as cp

        osc = cp.asarray(osc_dev).get()
        chg = cp.asarray(chg_dev).get()
    except Exception:
        
        pytest.skip("No CuPy; CUDA array interface consumer unavailable for host pull")

    
    osc_cpu, chg_cpu = ti.acosc(high.astype(np.float64), low.astype(np.float64))
    tol = 5e-4
    assert np.allclose(np.nan_to_num(osc, nan=0.0), np.nan_to_num(osc_cpu, nan=0.0), rtol=tol, atol=tol)
    assert np.allclose(np.nan_to_num(chg, nan=0.0), np.nan_to_num(chg_cpu, nan=0.0), rtol=tol, atol=tol)


@pytest.mark.skipif(not _has("acosc_cuda_many_series_one_param_dev"), reason="acosc CUDA bindings not present")
def test_acosc_cuda_many_series_one_param_shapes():
    
    rows, cols = 8, 128
    rng = np.random.default_rng(0)
    base = rng.normal(0, 1, size=(rows, cols)).astype(np.float32)
    high = base + 0.5
    low = base - 0.4
    osc_dev, chg_dev = ti.acosc_cuda_many_series_one_param_dev(high, low, device_id=0)

    
    assert hasattr(osc_dev, "__cuda_array_interface__")
    assert hasattr(chg_dev, "__cuda_array_interface__")
