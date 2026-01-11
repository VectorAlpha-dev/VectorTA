"""
Python binding tests for ATR CUDA kernels.
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
    if not hasattr(ti, 'atr_cuda_batch_dev'):
        return False
    try:
        
        x = np.arange(0, 64, dtype=np.float32)
        x[:5] = np.nan
        handle = ti.atr_cuda_batch_dev(x, x, x, (14, 14, 0))
        _ = cp.asarray(handle)
        return True
    except Exception as e:  
        msg = str(e).lower()
        if 'cuda not available' in msg or 'nvcc' in msg or 'ptx' in msg:
            return False
        return True


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available or cuda bindings not built")
class TestAtrCuda:
    def test_atr_cuda_batch_matches_cpu(self):
        data = load_test_data()
        high = data['high'].astype(np.float64)
        low = data['low'].astype(np.float64)
        close = data['close'].astype(np.float64)
        length = 14
        cpu = ti.atr(high, low, close, length)

        handle = ti.atr_cuda_batch_dev(high.astype(np.float32),
                                       low.astype(np.float32),
                                       close.astype(np.float32),
                                       (length, length, 0))
        gpu_row = cp.asnumpy(cp.asarray(handle))[0]

        assert_close(gpu_row, cpu, rtol=2e-3, atol=1e-5, msg="ATR CUDA batch vs CPU mismatch")

    def test_atr_cuda_many_series_one_param_matches_cpu(self):
        T = 2048
        N = 4
        base = load_test_data()['close'][:T].astype(np.float64)
        high_tm = np.zeros((T, N), dtype=np.float64)
        low_tm = np.zeros((T, N), dtype=np.float64)
        close_tm = np.zeros((T, N), dtype=np.float64)
        for j in range(N):
            off = 0.05 + 0.01 * j
            high_tm[:, j] = base + off
            low_tm[:, j] = base - off
            close_tm[:, j] = base

        length = 14
        cpu_tm = np.zeros_like(high_tm)
        for j in range(N):
            cpu_tm[:, j] = ti.atr(high_tm[:, j], low_tm[:, j], close_tm[:, j], length)

        handle = ti.atr_cuda_many_series_one_param_dev(
            high_tm.astype(np.float32).ravel(),
            low_tm.astype(np.float32).ravel(),
            close_tm.astype(np.float32).ravel(),
            N, T, length
        )
        gpu_tm = cp.asnumpy(cp.asarray(handle))

        assert gpu_tm.shape == cpu_tm.shape
        assert_close(gpu_tm, cpu_tm, rtol=2e-3, atol=1e-5, msg="ATR CUDA TM vs CPU mismatch")

