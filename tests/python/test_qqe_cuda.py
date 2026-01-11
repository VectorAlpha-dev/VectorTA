"""
Python binding tests for QQE CUDA kernels.
Skips gracefully when CUDA or bindings are unavailable.
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
    if not hasattr(ti, 'qqe_cuda_batch_dev'):
        return False
    try:
        x = np.array([np.nan, 1.0, 2.0, 3.0], dtype=np.float32)
        handle, meta = ti.qqe_cuda_batch_dev(
            x,
            rsi_period_range=(3, 3, 0),
            smoothing_factor_range=(2, 2, 0),
            fast_factor_range=(4.236, 4.236, 0.0),
        )
        _ = cp.asarray(handle)
        return True
    except Exception as e:
        msg = str(e).lower()
        if 'cuda not available' in msg or 'nvcc' in msg or 'ptx' in msg:
            return False
        return True


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available or cuda bindings not built")
class TestQqeCuda:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()

    def test_qqe_cuda_batch_matches_cpu(self, test_data):
        close = test_data['close']
        rsi_p, ema_p, fast_k = 14, 5, 4.236

        
        cpu_fast, cpu_slow = ti.qqe(close, rsi_p, ema_p, fast_k)

        
        handle, meta = ti.qqe_cuda_batch_dev(
            close.astype(np.float32),
            rsi_period_range=(rsi_p, rsi_p, 0),
            smoothing_factor_range=(ema_p, ema_p, 0),
            fast_factor_range=(fast_k, fast_k, 0.0),
        )
        gpu = cp.asnumpy(cp.asarray(handle))
        
        fast_row = gpu[0]
        slow_row = gpu[1]

        assert_close(fast_row, cpu_fast, rtol=5e-5, atol=1e-5, msg="QQE FAST CUDA vs CPU mismatch")
        assert_close(slow_row, cpu_slow, rtol=5e-5, atol=1e-5, msg="QQE SLOW CUDA vs CPU mismatch")

    def test_qqe_cuda_many_series_one_param_matches_cpu(self, test_data):
        T = 1024
        N = 4
        base = test_data['close'][:T].astype(np.float64)
        tm = np.zeros((T, N), dtype=np.float64)
        for j in range(N):
            tm[:, j] = base * (1.0 + 0.01 * j)

        rsi_p, ema_p, fast_k = 14, 5, 4.236

        
        cpu_fast = np.full_like(tm, np.nan)
        cpu_slow = np.full_like(tm, np.nan)
        for j in range(N):
            f, s = ti.qqe(tm[:, j], rsi_p, ema_p, fast_k)
            cpu_fast[:, j] = f
            cpu_slow[:, j] = s

        
        handle = ti.qqe_cuda_many_series_one_param_dev(
            tm.astype(np.float32), rsi_p, ema_p, fast_k
        )
        out_tm = cp.asnumpy(cp.asarray(handle))  
        gpu_fast = out_tm[:, :N]
        gpu_slow = out_tm[:, N:]

        assert_close(gpu_fast, cpu_fast, rtol=5e-5, atol=1e-5, msg="QQE FAST many-series CUDA vs CPU mismatch")
        assert_close(gpu_slow, cpu_slow, rtol=5e-5, atol=1e-5, msg="QQE SLOW many-series CUDA vs CPU mismatch")

