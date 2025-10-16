"""
Python binding tests for StdDev CUDA kernels.
Skips gracefully when CUDA/CuPy not available or CUDA feature not built.
"""
import pytest
import numpy as np

try:
    import cupy as cp
except ImportError:  # pragma: no cover
    cp = None

try:
    import my_project as ti
except ImportError:  # pragma: no cover
    pytest.skip(
        "Python module not built. Run 'maturin develop --features python,cuda' first",
        allow_module_level=True,
    )

from test_utils import assert_close, load_test_data


def _cuda_available() -> bool:
    if cp is None:
        return False
    if not hasattr(ti, "stddev_cuda_batch_dev"):
        return False
    try:
        x = np.array([np.nan, 1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        handle = ti.stddev_cuda_batch_dev(x, (3, 3, 0), (2.0, 2.0, 0.0))
        _ = cp.asarray(handle)
        return True
    except Exception as exc:  # pragma: no cover
        msg = str(exc).lower()
        if "cuda not available" in msg or "ptx" in msg or "nvcc" in msg:
            return False
        return True


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available or cuda bindings not built")
class TestStddevCuda:
    @pytest.fixture(scope="class")
    def test_data(self):
        return load_test_data()

    def test_stddev_cuda_batch_matches_cpu(self, test_data):
        close = test_data["close"][:4096].astype(np.float64)
        period_range = (8, 24, 8)
        nbdev_range = (0.5, 2.0, 0.5)

        cpu = ti.stddev_batch(close, period_range, nbdev_range)
        cpu_vals = cpu["values"].astype(np.float32)

        handle = ti.stddev_cuda_batch_dev(close.astype(np.float32), period_range, nbdev_range)
        gpu_vals = cp.asnumpy(cp.asarray(handle)).reshape(cpu_vals.shape)

        assert_close(gpu_vals, cpu_vals, rtol=1e-3, atol=1e-3, msg="CUDA stddev mismatch")

    def test_stddev_cuda_many_series_one_param_matches_cpu(self, test_data):
        T = 1024
        N = 4
        series = test_data["close"][:T].astype(np.float64)
        data_tm = np.zeros((T, N), dtype=np.float64)
        for j in range(N):
            data_tm[:, j] = series * (1.0 + 0.02 * j)

        period = 15
        nbdev = 2.0

        cpu_tm = np.zeros_like(data_tm)
        for j in range(N):
            cpu_tm[:, j] = ti.stddev(data_tm[:, j], period, nbdev)

        handle = ti.stddev_cuda_many_series_one_param_dev(
            data_tm.astype(np.float32).ravel(),  # time-major flattened
            N,
            T,
            period,
            nbdev,
        )
        gpu_tm = cp.asnumpy(cp.asarray(handle)).reshape((T, N))

        assert_close(gpu_tm, cpu_tm, rtol=1e-3, atol=1e-3, msg="CUDA stddev many-series mismatch")

