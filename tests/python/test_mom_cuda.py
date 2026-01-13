"""
Python binding tests for MOM CUDA kernels.
Skips gracefully when CUDA or CuPy is unavailable, or module not built.
"""
import numpy as np
import pytest

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
    if not hasattr(ti, "mom_cuda_batch_dev"):
        return False
    try:
        x = np.array([np.nan, 1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        handle = ti.mom_cuda_batch_dev(x, period_range=(2, 2, 0))
        _ = cp.asarray(handle)
        return True
    except Exception as exc:
        msg = str(exc).lower()
        if "cuda not available" in msg or "ptx" in msg or "nvcc" in msg:
            return False
        return True


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available or cuda bindings not built")
class TestMomCuda:
    @pytest.fixture(scope="class")
    def test_data(self):
        return load_test_data()

    def test_mom_cuda_batch_matches_cpu(self, test_data):
        close = test_data["close"][:4096].astype(np.float64)
        period = 14

        cpu = ti.mom(close, period)

        handle = ti.mom_cuda_batch_dev(close.astype(np.float32), period_range=(period, period, 0))
        gpu_row = cp.asnumpy(cp.asarray(handle))[0]

        assert gpu_row.shape == cpu.shape
        assert_close(gpu_row, cpu, rtol=2e-4, atol=5e-6, msg="MOM CUDA batch vs CPU mismatch")

    def test_mom_cuda_many_series_one_param_matches_cpu(self, test_data):
        T = 2048
        N = 4
        base = test_data["close"][:T].astype(np.float64)
        tm = np.zeros((T, N), dtype=np.float64)
        for j in range(N):
            tm[:, j] = base * (1.0 + 0.01 * (j + 1))

        period = 10
        cpu_tm = np.zeros_like(tm)
        for j in range(N):
            cpu_tm[:, j] = ti.mom(tm[:, j], period)

        handle = ti.mom_cuda_many_series_one_param_dev(
            tm.astype(np.float32).ravel(), N, T, period
        )
        gpu_tm = cp.asnumpy(cp.asarray(handle))
        assert gpu_tm.shape == cpu_tm.shape
        assert_close(gpu_tm, cpu_tm, rtol=2e-4, atol=5e-6, msg="MOM CUDA TM vs CPU mismatch")

