"""
CUDA bindings tests for the QStick indicator (average of close-open).
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
    pytest.skip(
        "Python module not built. Run 'maturin develop --features python,cuda' first",
        allow_module_level=True,
    )

from test_utils import load_test_data, assert_close


def _cuda_available() -> bool:
    if cp is None:
        return False
    if not hasattr(ti, "qstick_cuda_batch_dev"):
        return False
    try:
        
        o = np.array([np.nan, 1.0, 2.0, 3.0], dtype=np.float32)
        c = np.array([np.nan, 1.1, 1.9, 3.3], dtype=np.float32)
        handle = ti.qstick_cuda_batch_dev(o, c, period_range=(3, 3, 0))
        _ = cp.asarray(handle)
        return True
    except Exception as exc:  
        msg = str(exc).lower()
        if "cuda not available" in msg or "ptx" in msg or "driver" in msg:
            return False
        return True


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available or bindings missing")
class TestQstickCuda:
    @pytest.fixture(scope="class")
    def test_data(self):
        return load_test_data()

    def test_qstick_cuda_batch_matches_cpu(self, test_data):
        open_ = test_data["open"].astype(np.float64)
        close = test_data["close"].astype(np.float64)
        period = 14

        cpu = ti.qstick(open_, close, period)
        handle = ti.qstick_cuda_batch_dev(
            open_.astype(np.float32),
            close.astype(np.float32),
            period_range=(period, period, 0),
        )
        gpu_row = cp.asnumpy(cp.asarray(handle))[0]
        assert_close(gpu_row, cpu, rtol=1e-5, atol=1e-6, msg="CUDA QStick batch mismatch")

    def test_qstick_cuda_many_series_one_param_matches_cpu(self, test_data):
        T = 1024
        N = 4
        o_base = test_data["open"][:T].astype(np.float64)
        c_base = test_data["close"][:T].astype(np.float64)
        open_tm = np.zeros((T, N), dtype=np.float64)
        close_tm = np.zeros((T, N), dtype=np.float64)
        for j in range(N):
            open_tm[:, j] = o_base * (1.0 + 0.01 * j)
            close_tm[:, j] = c_base * (1.0 - 0.01 * j)

        period = 10

        cpu_tm = np.zeros_like(open_tm)
        for j in range(N):
            cpu_tm[:, j] = ti.qstick(open_tm[:, j], close_tm[:, j], period)

        handle = ti.qstick_cuda_many_series_one_param_dev(
            open_tm.astype(np.float32),
            close_tm.astype(np.float32),
            period,
        )
        gpu_tm = cp.asnumpy(cp.asarray(handle))

        assert gpu_tm.shape == open_tm.shape
        assert_close(gpu_tm, cpu_tm, rtol=1e-5, atol=1e-6, msg="CUDA QStick many-series mismatch")

