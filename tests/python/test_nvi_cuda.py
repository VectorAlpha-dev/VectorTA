"""
Python binding tests for NVI CUDA kernels.
Skips gracefully when CUDA or the CUDA bindings are unavailable.
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
    pytest.skip(
        "Python module not built. Run 'maturin develop --features python,cuda' first",
        allow_module_level=True,
    )

from test_utils import assert_close


def _cuda_available() -> bool:
    if cp is None:
        return False
    if not hasattr(ti, 'nvi_cuda_batch_dev'):
        return False
    try:
        n = 256
        close = np.ones(n, dtype=np.float32)
        volume = np.ones(n, dtype=np.float32)
        close[0] = 100.0
        volume[0] = 1000.0
        handle = ti.nvi_cuda_batch_dev(close, volume)
        _ = cp.asarray(handle)
        return True
    except Exception as e:
        msg = str(e).lower()
        if 'cuda not available' in msg or 'ptx' in msg or 'nvcc' in msg:
            return False
        return True


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available or cuda bindings not built")
class TestNviCuda:
    def test_nvi_cuda_batch_matches_cpu(self):
        n = 4096
        x = np.arange(n, dtype=np.float64)
        close = (np.sin(0.123 * 0.01 * x) + 0.0002 * x + 100.0).astype(np.float64)
        volume = (np.abs(np.cos(0.077 * 0.01 * x)) * 500.0 + 100.0).astype(np.float64)
        close[:3] = np.nan
        volume[:3] = np.nan

        cpu = ti.nvi(close, volume)

        handle = ti.nvi_cuda_batch_dev(close.astype(np.float32), volume.astype(np.float32))
        gpu = cp.asnumpy(cp.asarray(handle))[0]

        assert_close(gpu, cpu, rtol=2e-4, atol=5e-6, msg="NVI CUDA batch vs CPU mismatch")

    def test_nvi_cuda_many_series_one_param_matches_cpu(self):
        T = 2048
        N = 8
        x = np.arange(T, dtype=np.float64)
        base = np.sin(0.31 * 0.01 * x) * 0.0009 * x + 100.0
        close_tm = np.full((T, N), np.nan, dtype=np.float64)
        volume_tm = np.full((T, N), np.nan, dtype=np.float64)
        for j in range(N):
            start = min(j, 5)
            close_tm[start:, j] = base[start:] + 0.05 * np.sin(0.009 * x[start:])
            volume_tm[start:, j] = np.abs(np.cos(0.007 * x[start:])) * 400.0 + 120.0

        cpu_tm = np.zeros_like(close_tm)
        for j in range(N):
            cpu_tm[:, j] = ti.nvi(close_tm[:, j], volume_tm[:, j])

        handle = ti.nvi_cuda_many_series_one_param_dev(
            close_tm.astype(np.float32).ravel(),
            volume_tm.astype(np.float32).ravel(),
            N,
            T,
        )
        gpu_tm = cp.asnumpy(cp.asarray(handle))
        assert gpu_tm.shape == cpu_tm.shape
        assert_close(gpu_tm, cpu_tm, rtol=2e-4, atol=5e-6, msg="NVI CUDA TM vs CPU mismatch")

