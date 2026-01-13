"""
Python binding tests for DEC_OSC CUDA kernels.
Skips gracefully when CUDA is unavailable or bindings not present.
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
    if not hasattr(ti, 'dec_osc_cuda_batch_dev'):
        return False
    try:
        x = np.array([np.nan, 100.0, 101.0, 102.0, 103.0], dtype=np.float32)
        handle = ti.dec_osc_cuda_batch_dev(x, hp_period_range=(3, 3, 0), k_range=(1.0, 1.0, 0.0))
        _ = cp.asarray(handle)
        return True
    except Exception as e:
        msg = str(e).lower()
        if 'cuda not available' in msg or 'ptx' in msg:
            return False
        return True


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available or cuda bindings not built")
class TestDecOscCuda:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()

    def test_dec_osc_cuda_batch_matches_cpu(self, test_data):
        close = test_data['close'].astype(np.float64)
        p, k = 64, 1.0
        cpu = ti.dec_osc(close, p, k)
        handle = ti.dec_osc_cuda_batch_dev(close.astype(np.float32), (p, p, 0), (k, k, 0.0))
        gpu = cp.asnumpy(cp.asarray(handle))[0]
        assert_close(gpu, cpu, rtol=3e-3, atol=1e-4, msg="CUDA batch vs CPU mismatch")

    def test_dec_osc_cuda_many_series_one_param_matches_cpu(self, test_data):
        T, N = 1024, 4
        series = test_data['close'][:T].astype(np.float64)
        data_tm = np.zeros((T, N), dtype=np.float64)
        for j in range(N):
            data_tm[:, j] = series * (1.0 + 0.01 * j)
        p, k = 32, 1.0
        cpu_tm = np.zeros_like(data_tm)
        for j in range(N):
            cpu_tm[:, j] = ti.dec_osc(data_tm[:, j], p, k)
        handle = ti.dec_osc_cuda_many_series_one_param_dev(data_tm.astype(np.float32).ravel(), N, T, p, k)
        gpu_tm = cp.asnumpy(cp.asarray(handle)).reshape(T, N)
        assert_close(gpu_tm, cpu_tm, rtol=3e-3, atol=1e-4, msg="CUDA many-series vs CPU mismatch")

