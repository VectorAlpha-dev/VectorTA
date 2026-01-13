"""
Python binding tests for VPT CUDA kernels.
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
    if not hasattr(ti, 'vpt_cuda_batch_dev'):
        return False
    try:
        n = 256
        price = np.full(n, np.nan, dtype=np.float32)
        volume = np.full(n, np.nan, dtype=np.float32)
        price[0] = 100.0
        price[1] = 100.1
        volume[1] = 500.0
        handle = ti.vpt_cuda_batch_dev(price, volume)
        _ = cp.asarray(handle)
        return True
    except Exception as e:
        msg = str(e).lower()
        if 'cuda not available' in msg or 'ptx' in msg or 'nvcc' in msg:
            return False
        return True


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available or cuda bindings not built")
class TestVptCuda:
    def test_vpt_cuda_batch_matches_cpu(self):
        n = 4096
        x = np.arange(n, dtype=np.float64)
        price = np.full(n, np.nan, dtype=np.float64)
        volume = np.full(n, np.nan, dtype=np.float64)
        for i in range(4, n):
            price[i] = np.sin(0.0121 * x[i]) * 0.001 + 0.00019 * x[i] + 100.0
            volume[i] = np.abs(np.cos(0.0081 * x[i])) * 600.0 + 150.0

        cpu = ti.vpt(price, volume)

        handle = ti.vpt_cuda_batch_dev(price.astype(np.float32), volume.astype(np.float32))
        gpu = cp.asnumpy(cp.asarray(handle)).reshape(1, n)[0]

        assert_close(gpu, cpu, rtol=2e-4, atol=5e-6, msg="VPT CUDA batch vs CPU mismatch")

    def test_vpt_cuda_many_series_one_param_matches_cpu(self):
        T = 2048
        N = 8
        x = np.arange(T, dtype=np.float64)
        price_tm = np.full((T, N), np.nan, dtype=np.float64)
        volume_tm = np.full((T, N), np.nan, dtype=np.float64)
        for j in range(N):
            start = min(j, 6)
            price_tm[start:, j] = np.sin(0.013 * x[start:]) * 0.0007 * x[start:] + 100.0
            volume_tm[start:, j] = np.abs(np.cos(0.011 * x[start:])) * 450.0 + 110.0
            price_tm[start, j] = 100.0

        cpu_tm = np.zeros_like(price_tm)
        for j in range(N):
            cpu_tm[:, j] = ti.vpt(price_tm[:, j], volume_tm[:, j])

        handle = ti.vpt_cuda_many_series_one_param_dev(
            price_tm.astype(np.float32).ravel(),
            volume_tm.astype(np.float32).ravel(),
            N,
            T,
        )
        gpu_tm = cp.asnumpy(cp.asarray(handle)).reshape(T, N)
        assert gpu_tm.shape == cpu_tm.shape
        assert_close(gpu_tm, cpu_tm, rtol=2e-4, atol=5e-6, msg="VPT CUDA TM vs CPU mismatch")

