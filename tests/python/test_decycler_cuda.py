"""
Python binding tests for Decycler CUDA kernels.
Skips gracefully when CUDA or CuPy is unavailable or bindings not built.
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
    if not hasattr(ti, "decycler_cuda_batch_dev"):
        return False
    try:
        x = np.array([np.nan, 1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        handle = ti.decycler_cuda_batch_dev(
            x, hp_period_range=(3, 3, 0), k_range=(0.707, 0.707, 0.0)
        )
        _ = cp.asarray(handle)
        return True
    except Exception as exc:
        msg = str(exc).lower()
        if "cuda not available" in msg or "ptx" in msg or "nvcc" in msg:
            return False
        return True


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available or cuda bindings not built")
class TestDecyclerCuda:
    @pytest.fixture(scope="class")
    def test_data(self):
        return load_test_data()

    def test_decycler_cuda_batch_matches_cpu(self, test_data):
        close = test_data["close"][:4096].astype(np.float64)
        hp_range = (6, 32, 5)
        k_range = (0.3, 0.9, 0.2)

        cpu = ti.decycler_batch(close, hp_range, k_range)
        rows, cols = cpu["rows"], cpu["cols"]
        cpu_vals = cpu["values"].reshape(rows, cols)

        handle = ti.decycler_cuda_batch_dev(
            close.astype(np.float32), hp_range, k_range
        )
        gpu = cp.asnumpy(cp.asarray(handle)).reshape(rows, cols)

        assert gpu.shape == cpu_vals.shape
        assert_close(gpu, cpu_vals, rtol=2e-5, atol=2e-5, msg="CUDA batch vs CPU mismatch")

    def test_decycler_cuda_many_series_one_param_matches_cpu(self, test_data):
        T, N = 2048, 4
        base = test_data["close"][:T].astype(np.float64)
        data_tm = np.zeros((T, N), dtype=np.float64)
        for j in range(N):
            data_tm[:, j] = base * (1.0 + 0.01 * j)

        hp_period = 48
        k = 0.707

        cpu_tm = np.zeros_like(data_tm)
        for j in range(N):
            cpu_tm[:, j] = ti.decycler(data_tm[:, j], hp_period=hp_period, k=k)

        handle = ti.decycler_cuda_many_series_one_param_dev(
            data_tm.astype(np.float32), hp_period, k
        )
        gpu_tm = cp.asnumpy(cp.asarray(handle))

        assert gpu_tm.shape == data_tm.shape
        assert_close(gpu_tm, cpu_tm, rtol=2e-5, atol=2e-5, msg="CUDA many-series vs CPU mismatch")

