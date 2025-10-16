"""
Python binding tests for TSI CUDA kernels.
Mirrors CPU behavior and ALMA-style CUDA API. Skips cleanly if CUDA unavailable.
"""
import pytest
import numpy as np

try:
    import cupy as cp
except ImportError:  # optional dependency
    cp = None

try:
    import my_project as ti
except ImportError:
    pytest.skip("Python module not built. Run 'maturin develop --features python,cuda' first", allow_module_level=True)

from test_utils import load_test_data, assert_close


def _cuda_available() -> bool:
    if cp is None:
        return False
    if not hasattr(ti, 'tsi_cuda_batch_dev'):
        return False
    try:
        probe = np.array([np.nan, 1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        handle, meta = ti.tsi_cuda_batch_dev(
            probe, (10, 10, 0), (5, 5, 0)
        )
        _ = cp.asarray(handle)
        return True
    except Exception as e:
        msg = str(e).lower()
        if 'cuda not available' in msg or 'ptx' in msg or 'nvcc' in msg:
            return False
        return True


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available or bindings missing")
class TestTsiCuda:
    @pytest.fixture(scope='class')
    def test_data(self):
        return load_test_data()

    def test_tsi_cuda_batch_matches_cpu(self, test_data):
        close = test_data['close']
        long_p, short_p = 25, 13

        cpu = ti.tsi(close, long_p, short_p)

        handle, meta = ti.tsi_cuda_batch_dev(
            close.astype(np.float32),
            (long_p, long_p, 0),
            (short_p, short_p, 0),
        )
        gpu = cp.asnumpy(cp.asarray(handle))[0]

        assert_close(gpu, cpu, rtol=2e-3, atol=2e-4, msg="TSI CUDA batch vs CPU mismatch")

    def test_tsi_cuda_many_series_one_param_matches_cpu(self, test_data):
        T, N = 2048, 4
        base = test_data['close'][:T].astype(np.float64)
        data_tm = np.zeros((T, N), dtype=np.float64)
        for j in range(N):
            data_tm[1:, j] = base[1:] * (1.0 + 0.01 * j)  # ensure prev exists for momentum

        long_p, short_p = 25, 13

        cpu_tm = np.zeros_like(data_tm)
        for j in range(N):
            cpu_tm[:, j] = ti.tsi(data_tm[:, j], long_p, short_p)

        handle = ti.tsi_cuda_many_series_one_param_dev(
            data_tm.astype(np.float32), data_tm.shape[1], data_tm.shape[0], long_p, short_p
        )
        gpu_tm = cp.asnumpy(cp.asarray(handle))
        assert gpu_tm.shape == data_tm.shape
        assert_close(gpu_tm, cpu_tm, rtol=2e-3, atol=2e-4, msg="TSI CUDA many-series vs CPU mismatch")

