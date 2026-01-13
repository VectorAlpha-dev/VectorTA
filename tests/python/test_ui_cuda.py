"""
Python binding tests for UI (Ulcer Index) CUDA kernels.
Skips gracefully when CUDA or CuPy is unavailable.
"""
import pytest
import numpy as np

try:
    import cupy as cp
except ImportError:
    cp = None

try:
    import my_project as ti
except ImportError:
    pytest.skip("Python module not built. Run 'maturin develop --features python,cuda' first", allow_module_level=True)

from test_utils import assert_close


def _cuda_available() -> bool:
    if cp is None:
        return False
    if not hasattr(ti, 'ui_cuda_batch_dev'):
        return False
    try:
        x = np.array([np.nan, 1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        handle = ti.ui_cuda_batch_dev(
            x,
            period_range=(3, 3, 0),
            scalar_range=(100.0, 100.0, 0.0),
        )
        _ = cp.asarray(handle)
        return True
    except Exception as e:
        msg = str(e).lower()
        if 'cuda not available' in msg or 'nvcc' in msg or 'ptx' in msg:
            return False
        return True


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available or cuda bindings not built")
class TestUiCuda:
    def test_ui_cuda_batch_matches_cpu(self):

        n = 4096
        x = np.full(n, np.nan, dtype=np.float64)
        for i in range(8, n):
            t = float(i)
            x[i] = np.sin(t * 0.0017) + 0.0002 * t

        period = 14
        scalar = 100.0


        cpu = ti.ui(x, period, scalar)


        handle = ti.ui_cuda_batch_dev(
            x.astype(np.float32),
            period_range=(period, period, 0),
            scalar_range=(scalar, scalar, 0.0),
        )
        gpu = cp.asnumpy(cp.asarray(handle))[0]

        assert_close(gpu, cpu, rtol=2e-3, atol=2e-3, msg="UI CUDA batch vs CPU mismatch")

    def test_ui_cuda_many_series_one_param_matches_cpu(self):
        T = 2048
        N = 4
        base = np.zeros((T,), dtype=np.float64)
        for i in range(T):
            base[i] = np.sin(i * 0.0013) + 0.0003 * i
        tm = np.zeros((T, N), dtype=np.float64)
        for j in range(N):
            tm[:, j] = base * (1.0 + 0.01 * j)

        period = 14
        scalar = 100.0


        cpu_tm = np.zeros_like(tm)
        for j in range(N):
            cpu_tm[:, j] = ti.ui(tm[:, j], period, scalar)

        handle = ti.ui_cuda_many_series_one_param_dev(
            tm.astype(np.float32), period, scalar
        )
        gpu_tm = cp.asnumpy(cp.asarray(handle))
        assert gpu_tm.shape == tm.shape
        assert_close(gpu_tm, cpu_tm, rtol=2e-3, atol=2e-3, msg="UI CUDA many-series vs CPU mismatch")

