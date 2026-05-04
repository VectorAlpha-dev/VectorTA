import numpy as np
import pytest

cp = pytest.importorskip("cupy")

try:
    import vector_ta as ti
except Exception:
    ti = None


@pytest.mark.skipif(ti is None or not hasattr(ti, "rsmk_cuda_batch_dev"), reason="CUDA or RSMK CUDA binding unavailable")
def test_rsmk_cuda_batch_shapes_and_basic():
    n = 4096
    x = np.full(n, np.nan, dtype=np.float32)
    y = np.full(n, np.nan, dtype=np.float32)
    for i in range(12, n):
        t = float(i)
        x[i] = np.sin(0.0013 * t) + 0.0002 * t
        y[i] = np.abs(np.cos(0.0009 * t)) + 0.5

    ind, sig = ti.rsmk_cuda_batch_dev(x, y, (30, 30, 0), (3, 3, 0), (20, 20, 0))
    assert cp.asarray(ind).shape == (1, n)
    assert cp.asarray(sig).shape == (1, n)


@pytest.mark.skipif(ti is None or not hasattr(ti, "rsmk_cuda_many_series_one_param_dev"), reason="CUDA or RSMK CUDA binding unavailable")
def test_rsmk_cuda_many_series_shape():
    cols, rows = 8, 2048
    tm_main = np.full((rows, cols), np.nan, dtype=np.float32)
    tm_comp = np.full((rows, cols), np.nan, dtype=np.float32)
    for s in range(cols):
        for r in range(s, rows):
            t = float(r) + 0.2 * float(s)
            tm_main[r, s] = np.sin(0.002 * t) + 0.0003 * t
            tm_comp[r, s] = np.abs(np.cos(0.001 * t)) + 0.4

    ind, sig = ti.rsmk_cuda_many_series_one_param_dev(
        tm_main, tm_comp, cols, rows, 30, 3, 10
    )
    assert cp.asarray(ind).shape == (rows, cols)
    assert cp.asarray(sig).shape == (rows, cols)
