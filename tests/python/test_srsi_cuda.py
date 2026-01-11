"""
Python binding tests for SRSI CUDA kernels.
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
    if not hasattr(ti, 'srsi_cuda_batch_dev'):
        return False
    try:
        x = np.arange(0, 128, dtype=np.float32)
        x[:16] = np.nan
        d = ti.srsi_cuda_batch_dev(x, (14, 14, 0), (14, 14, 0), (3, 3, 0), (3, 3, 0))
        _ = cp.asarray(d["k"])  
        return True
    except Exception as e:
        msg = str(e).lower()
        if 'cuda not available' in msg or 'nvcc' in msg or 'ptx' in msg:
            return False
        return True


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available or cuda bindings not built")
class TestSrsiCuda:
    def test_srsi_cuda_batch_matches_cpu(self):
        close = load_test_data()['close'].astype(np.float64)
        rp, sp, k, d = 14, 14, 3, 3
        cpu_k, cpu_d = ti.srsi(close, rsi_period=rp, stoch_period=sp, k=k, d=d)

        out = ti.srsi_cuda_batch_dev(close.astype(np.float32), (rp, rp, 0), (sp, sp, 0), (k, k, 0), (d, d, 0))
        gk = cp.asnumpy(cp.asarray(out["k"]))[0]
        gd = cp.asnumpy(cp.asarray(out["d"]))[0]

        assert_close(gk, cpu_k, rtol=1e-3, atol=5e-3, msg="SRSI K CUDA batch vs CPU mismatch")
        assert_close(gd, cpu_d, rtol=1e-3, atol=5e-3, msg="SRSI D CUDA batch vs CPU mismatch")

    def test_srsi_cuda_many_series_one_param_matches_cpu(self):
        T = 1024
        N = 6
        base = load_test_data()['close'][:T].astype(np.float64)
        tm = np.zeros((T, N), dtype=np.float64)
        for j in range(N):
            off = 0.01 * (j + 1)
            tm[:, j] = base + off

        rp, sp, k, d = 14, 14, 3, 3
        cpu_k = np.zeros_like(tm)
        cpu_d = np.zeros_like(tm)
        for j in range(N):
            kj, dj = ti.srsi(tm[:, j], rsi_period=rp, stoch_period=sp, k=k, d=d)
            cpu_k[:, j] = kj
            cpu_d[:, j] = dj

        out = ti.srsi_cuda_many_series_one_param_dev(tm.astype(np.float32), rp, sp, k, d)
        rows = out['rows']; cols = out['cols']
        gk = cp.asnumpy(cp.asarray(out['k'])).reshape(rows, cols)
        gd = cp.asnumpy(cp.asarray(out['d'])).reshape(rows, cols)

        assert gk.shape == cpu_k.shape and gd.shape == cpu_d.shape
        assert_close(gk, cpu_k, rtol=1e-3, atol=5e-3, msg="SRSI K CUDA TM vs CPU mismatch")
        assert_close(gd, cpu_d, rtol=1e-3, atol=5e-3, msg="SRSI D CUDA TM vs CPU mismatch")

