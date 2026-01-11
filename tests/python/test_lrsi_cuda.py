"""
Python binding tests for LRSI CUDA kernels.
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

from test_utils import load_test_data, assert_close


def _cuda_available() -> bool:
    if cp is None:
        return False
    if not hasattr(ti, 'lrsi_cuda_batch_dev'):
        return False
    try:
        high = np.array([np.nan, 2.0, 2.2, 2.1, 2.3], dtype=np.float32)
        low = np.array([np.nan, 1.8, 2.0, 1.9, 2.1], dtype=np.float32)
        handle = ti.lrsi_cuda_batch_dev(high, low, (0.2, 0.2, 0.0))
        _ = cp.asarray(handle)
        return True
    except Exception as e:
        msg = str(e).lower()
        if 'cuda not available' in msg or 'ptx' in msg:
            return False
        return True


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available or CUDA bindings not built")
class TestLrsiCuda:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()

    def test_lrsi_cuda_batch_matches_cpu(self, test_data):
        high = test_data['high']
        low = test_data['low']
        alpha = 0.2
        
        cpu = ti.lrsi(high, low, alpha)
        
        handle = ti.lrsi_cuda_batch_dev(high.astype(np.float32), low.astype(np.float32), (alpha, alpha, 0.0))
        gpu = cp.asnumpy(cp.asarray(handle))[0]
        assert_close(gpu, cpu, rtol=1e-5, atol=1e-5, msg="LRSI CUDA batch vs CPU mismatch")

    def test_lrsi_cuda_many_series_one_param_matches_cpu(self, test_data):
        T = 2048
        N = 4
        high = test_data['high'][:T].astype(np.float64)
        low = test_data['low'][:T].astype(np.float64)
        high_tm = np.zeros((T, N), dtype=np.float64)
        low_tm = np.zeros((T, N), dtype=np.float64)
        for j in range(N):
            scale = 1.0 + 0.01 * j
            high_tm[:, j] = high * scale
            low_tm[:, j] = low * scale
        alpha = 0.2
        
        cpu_tm = np.zeros_like(high_tm)
        for j in range(N):
            cpu_tm[:, j] = ti.lrsi(high_tm[:, j], low_tm[:, j], alpha)
        
        handle = ti.lrsi_cuda_many_series_one_param_dev(
            high_tm.astype(np.float32), low_tm.astype(np.float32), alpha
        )
        gpu_tm = cp.asnumpy(cp.asarray(handle))
        assert gpu_tm.shape == cpu_tm.shape
        assert_close(gpu_tm, cpu_tm, rtol=1e-5, atol=1e-5, msg="LRSI CUDA many-series vs CPU mismatch")

