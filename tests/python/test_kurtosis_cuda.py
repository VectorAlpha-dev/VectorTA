"""
Python binding tests for Kurtosis CUDA kernels.
Skips gracefully when CUDA/CuPy not available or CUDA feature not built.
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
    pytest.skip(
        "Python module not built. Run 'maturin develop --features python,cuda' first",
        allow_module_level=True,
    )

from test_utils import assert_close, load_test_data


def _cuda_available() -> bool:
    if cp is None:
        return False
    if not hasattr(ti, "kurtosis_cuda_batch_dev"):
        return False
    try:
        x = np.array([np.nan, 1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        handle = ti.kurtosis_cuda_batch_dev(x, (3, 3, 0))
        _ = cp.asarray(handle)
        return True
    except Exception as exc:  
        msg = str(exc).lower()
        if "cuda not available" in msg or "ptx" in msg or "nvcc" in msg:
            return False
        return True


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available or cuda bindings not built")
class TestKurtosisCuda:
    @pytest.fixture(scope="class")
    def test_data(self):
        return load_test_data()

    def test_kurtosis_cuda_batch_matches_cpu(self, test_data):
        close = test_data["close"][:4096].astype(np.float64)
        period_range = (8, 24, 8)

        cpu = ti.kurtosis_batch(close, period_range)
        cpu_vals = cpu["values"].astype(np.float32)

        handle = ti.kurtosis_cuda_batch_dev(close.astype(np.float32), period_range)
        gpu_vals = cp.asnumpy(cp.asarray(handle)).reshape(cpu_vals.shape)

        assert_close(gpu_vals, cpu_vals, rtol=1e-3, atol=1e-3, msg="CUDA kurtosis mismatch")

    def test_kurtosis_cuda_many_series_one_param_matches_cpu(self, test_data):
        T = 1024
        N = 4
        series = test_data["close"][:T].astype(np.float64)
        data_tm = np.zeros((T, N), dtype=np.float64)
        for j in range(N):
            data_tm[:, j] = series * (1.0 + 0.02 * j)

        period = 15

        cpu_tm = np.zeros_like(data_tm)
        for j in range(N):
            cpu_tm[:, j] = ti.kurtosis(data_tm[:, j], period)

        handle = ti.kurtosis_cuda_many_series_one_param_dev(data_tm.astype(np.float32), period)
        gpu_tm = cp.asnumpy(cp.asarray(handle))

        assert gpu_tm.shape == data_tm.shape
        assert_close(gpu_tm, cpu_tm, rtol=1e-3, atol=1e-3, msg="CUDA kurtosis many-series mismatch")

