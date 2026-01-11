"""
Python binding tests for VAR CUDA kernels (rolling variance with nbdev scaling).
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
    if not hasattr(ti, 'var_cuda_batch_dev'):
        return False
    try:
        x = np.array([np.nan, 1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        h = ti.var_cuda_batch_dev(x.astype(np.float32), period_range=(3, 3, 0), nbdev_range=(1.0, 1.0, 0.0))
        _ = cp.asarray(h)
        return True
    except Exception as e:
        msg = str(e).lower()
        if 'cuda not available' in msg or 'nvcc' in msg or 'ptx' in msg:
            return False
        return True


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available or cuda bindings not built")
class TestVarCuda:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()

    def test_var_cuda_batch_matches_cpu(self, test_data):
        close = test_data['close']
        period, nbdev = 14, 1.5

        
        cpu = ti.var(close, period, nbdev)

        
        handle = ti.var_cuda_batch_dev(
            close.astype(np.float32),
            period_range=(period, period, 0),
            nbdev_range=(nbdev, nbdev, 0.0),
        )
        gpu = cp.asarray(handle)
        gpu_first = cp.asnumpy(gpu)[0]

        
        assert_close(gpu_first, cpu, rtol=5e-4, atol=1e-5, msg="VAR CUDA batch vs CPU mismatch")

    def test_var_cuda_many_series_one_param_matches_cpu(self, test_data):
        T, N = 1024, 4
        series = test_data['close'][:T].astype(np.float64)
        data_tm = np.zeros((T, N), dtype=np.float64)
        for j in range(N):
            data_tm[:, j] = series * (1.0 + 0.01 * j)

        period, nbdev = 14, 1.0

        
        cpu_tm = np.zeros_like(data_tm)
        for j in range(N):
            cpu_tm[:, j] = ti.var(data_tm[:, j], period, nbdev)

        
        handle = ti.var_cuda_many_series_one_param_dev(
            data_tm.astype(np.float32).ravel(), cols=N, rows=T, period=period, nbdev=nbdev
        )
        gpu_tm = cp.asnumpy(cp.asarray(handle))
        assert gpu_tm.shape == (T, N)
        assert_close(gpu_tm, cpu_tm, rtol=5e-4, atol=1e-5, msg="VAR CUDA many-series vs CPU mismatch")

