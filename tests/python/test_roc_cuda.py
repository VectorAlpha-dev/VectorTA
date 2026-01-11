"""
Python binding tests for ROC CUDA kernels.
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
    pytest.skip("Python module not built. Run 'maturin develop --features python,cuda' first", allow_module_level=True)

from test_utils import load_test_data, assert_close


def _cuda_available() -> bool:
    if cp is None:
        return False
    if not hasattr(ti, 'roc_cuda_batch_dev'):
        return False
    try:
        x = np.arange(0, 128, dtype=np.float32)
        x[:10] = np.nan  
        handle = ti.roc_cuda_batch_dev(x, (10, 10, 0))
        _ = cp.asarray(handle)
        return True
    except Exception as e:
        msg = str(e).lower()
        if 'cuda not available' in msg or 'nvcc' in msg or 'ptx' in msg:
            return False
        return True


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available or cuda bindings not built")
class TestRocCuda:
    def test_roc_cuda_batch_matches_cpu(self):
        close = load_test_data()['close'].astype(np.float64)
        period = 14
        cpu = ti.roc(close, period=period)

        handle = ti.roc_cuda_batch_dev(close.astype(np.float32), (period, period, 0))
        gpu_row = cp.asnumpy(cp.asarray(handle))[0]

        
        assert_close(gpu_row, cpu, rtol=1e-3, atol=5e-4, msg="ROC CUDA batch vs CPU mismatch")

    def test_roc_cuda_many_series_one_param_matches_cpu(self):
        T = 2048
        N = 5
        base = load_test_data()['close'][:T].astype(np.float64)
        tm = np.zeros((T, N), dtype=np.float64)
        for j in range(N):
            off = 0.02 * (j + 1)
            tm[:, j] = base + off

        period = 10
        cpu_tm = np.zeros_like(tm)
        for j in range(N):
            cpu_tm[:, j] = ti.roc(tm[:, j], period=period)

        handle = ti.roc_cuda_many_series_one_param_dev(tm.astype(np.float32).ravel(), N, T, period)
        gpu_tm = cp.asnumpy(cp.asarray(handle)).reshape(T, N)

        assert gpu_tm.shape == cpu_tm.shape
        assert_close(gpu_tm, cpu_tm, rtol=1e-3, atol=5e-4, msg="ROC CUDA TM vs CPU mismatch")

