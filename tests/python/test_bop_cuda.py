"""
Python binding tests for BOP CUDA kernels.
Skips gracefully when CUDA or the CUDA bindings are unavailable.
"""
import pytest
import numpy as np

try:
    import cupy as cp
except ImportError:  # pragma: no cover
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
    if not hasattr(ti, 'bop_cuda_batch_dev'):
        return False
    try:
        n = 256
        open = np.zeros(n, dtype=np.float32)
        high = np.ones(n, dtype=np.float32) * 2
        low = np.zeros(n, dtype=np.float32)
        close = np.ones(n, dtype=np.float32)
        handle = ti.bop_cuda_batch_dev(open, high, low, close)
        _ = cp.asarray(handle)
        return True
    except Exception as e:
        msg = str(e).lower()
        if 'cuda not available' in msg or 'ptx' in msg or 'nvcc' in msg:
            return False
        return True


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available or cuda bindings not built")
class TestBopCuda:
    def test_bop_cuda_batch_matches_cpu(self):
        n = 8192
        x = np.arange(n, dtype=np.float64)
        open = np.sin(0.13 * x) * 0.001 * x + 0.1
        high = open + (0.4 + 0.03 * np.cos(0.01 * x)).astype(np.float64)
        low = open - (0.39 + 0.02 * np.sin(0.01 * x)).astype(np.float64)
        close = open + 0.1 * np.sin(0.7 * 0.01 * x)

        cpu = ti.bop(open, high, low, close)

        handle = ti.bop_cuda_batch_dev(
            open.astype(np.float32),
            high.astype(np.float32),
            low.astype(np.float32),
            close.astype(np.float32),
        )
        gpu = cp.asnumpy(cp.asarray(handle))[0]

        assert_close(gpu, cpu, rtol=2e-4, atol=5e-6, msg="BOP CUDA batch vs CPU mismatch")

    def test_bop_cuda_many_series_one_param_matches_cpu(self):
        T = 2048
        N = 4
        x = np.arange(T, dtype=np.float64)
        base = np.sin(0.31 * x) * 0.0009 * x
        open_tm = np.zeros((T, N), dtype=np.float64)
        high_tm = np.zeros((T, N), dtype=np.float64)
        low_tm = np.zeros((T, N), dtype=np.float64)
        close_tm = np.zeros((T, N), dtype=np.float64)
        for j in range(N):
            off = 0.3 + 0.02 * (j + 1) / N
            open_tm[:, j] = base + 0.001 * np.cos(0.01 * x)
            high_tm[:, j] = base + off + 0.01 * np.sin(0.01 * x)
            low_tm[:, j] = base - off - 0.01 * np.cos(0.01 * x)
            close_tm[:, j] = base + 0.05 * np.sin(0.009 * x)

        cpu_tm = np.zeros_like(open_tm)
        for j in range(N):
            cpu_tm[:, j] = ti.bop(
                open_tm[:, j].astype(np.float64),
                high_tm[:, j].astype(np.float64),
                low_tm[:, j].astype(np.float64),
                close_tm[:, j].astype(np.float64),
            )

        handle = ti.bop_cuda_many_series_one_param_dev(
            open_tm.astype(np.float32).ravel(),
            high_tm.astype(np.float32).ravel(),
            low_tm.astype(np.float32).ravel(),
            close_tm.astype(np.float32).ravel(),
            N,
            T,
        )
        gpu_tm = cp.asnumpy(cp.asarray(handle))
        assert gpu_tm.shape == cpu_tm.shape
        assert_close(gpu_tm, cpu_tm, rtol=2e-4, atol=5e-6, msg="BOP CUDA TM vs CPU mismatch")

