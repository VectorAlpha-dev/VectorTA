"""
Python binding tests for StochF CUDA kernels.
Skips gracefully when CUDA is unavailable or CUDA feature not built.
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
    if not hasattr(ti, 'stochf_cuda_batch_dev'):
        return False
    try:

        x = np.array([np.nan, 1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        k_h, d_h = ti.stochf_cuda_batch_dev(
            x.astype(np.float32), x.astype(np.float32), x.astype(np.float32),
            fastk_range=(3, 3, 0), fastd_range=(3, 3, 0)
        )
        _ = cp.asarray(k_h); _ = cp.asarray(d_h)
        return True
    except Exception as e:
        msg = str(e).lower()
        if 'cuda not available' in msg or 'nvcc' in msg or 'ptx' in msg:
            return False
        return True


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available or cuda bindings not built")
class TestStochfCuda:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()

    def test_stochf_cuda_batch_matches_cpu(self, test_data):
        high = test_data['high']; low = test_data['low']; close = test_data['close']
        fk, fd = 5, 3


        k_cpu, d_cpu = ti.stochf(high, low, close, fk, fd, 0)


        k_h, d_h = ti.stochf_cuda_batch_dev(
            high.astype(np.float32), low.astype(np.float32), close.astype(np.float32),
            fastk_range=(fk, fk, 0), fastd_range=(fd, fd, 0)
        )
        k_gpu = cp.asnumpy(cp.asarray(k_h))[0]
        d_gpu = cp.asnumpy(cp.asarray(d_h))[0]

        assert_close(k_gpu, k_cpu, rtol=1e-4, atol=2e-4, msg="CUDA K vs CPU mismatch")
        assert_close(d_gpu, d_cpu, rtol=1e-4, atol=2e-4, msg="CUDA D vs CPU mismatch")

    def test_stochf_cuda_many_series_one_param_matches_cpu(self, test_data):
        T = 1024; N = 4
        base_h = test_data['high'][:T].astype(np.float64)
        base_l = test_data['low'][:T].astype(np.float64)
        base_c = test_data['close'][:T].astype(np.float64)
        high_tm = np.zeros((T, N), dtype=np.float64)
        low_tm  = np.zeros((T, N), dtype=np.float64)
        close_tm= np.zeros((T, N), dtype=np.float64)
        for j in range(N):
            s = 1.0 + 0.01 * j
            high_tm[:, j]  = base_h * s
            low_tm[:, j]   = base_l * s
            close_tm[:, j] = base_c * s

        fk, fd = 9, 3

        k_cpu_tm = np.zeros_like(close_tm)
        d_cpu_tm = np.zeros_like(close_tm)
        for j in range(N):
            k_cpu_tm[:, j], d_cpu_tm[:, j] = ti.stochf(high_tm[:, j], low_tm[:, j], close_tm[:, j], fk, fd, 0)

        k_h, d_h = ti.stochf_cuda_many_series_one_param_dev(
            high_tm.astype(np.float32), low_tm.astype(np.float32), close_tm.astype(np.float32),
            cols=N, rows=T, fastk=fk, fastd=fd, fastd_matype=0
        )
        k_gpu_tm = cp.asnumpy(cp.asarray(k_h))
        d_gpu_tm = cp.asnumpy(cp.asarray(d_h))
        assert k_gpu_tm.shape == (T, N)
        assert d_gpu_tm.shape == (T, N)
        assert_close(k_gpu_tm, k_cpu_tm, rtol=1e-4, atol=2e-4, msg="CUDA many-series K vs CPU mismatch")
        assert_close(d_gpu_tm, d_cpu_tm, rtol=1e-4, atol=2e-4, msg="CUDA many-series D vs CPU mismatch")

