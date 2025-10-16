"""CUDA bindings tests for ADXR (Average Directional Index Rating)."""
import pytest
import numpy as np

try:
    import cupy as cp
except ImportError:  # pragma: no cover - optional dependency
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
    if not hasattr(ti, "adxr_cuda_batch_dev"):
        return False
    try:
        # Probe a tiny call and ensure we can form a CuPy view
        arr = np.array([np.nan, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        handle = ti.adxr_cuda_batch_dev(arr, arr, arr, (3, 3, 0))
        _ = cp.asarray(handle)
        return True
    except Exception as exc:  # pragma: no cover - probing path
        msg = str(exc).lower()
        if "cuda not available" in msg or "ptx" in msg or "driver" in msg:
            return False
        return True


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available or cuda bindings missing")
class TestAdxrCuda:
    @pytest.fixture(scope="class")
    def test_data(self):
        return load_test_data()

    def test_adxr_cuda_batch_matches_cpu(self, test_data):
        high = test_data["high"].astype(np.float64)
        low = test_data["low"].astype(np.float64)
        close = test_data["close"].astype(np.float64)
        period = 14

        cpu = ti.adxr(high, low, close, period)

        handle = ti.adxr_cuda_batch_dev(
            high.astype(np.float32),
            low.astype(np.float32),
            close.astype(np.float32),
            (period, period, 0),
        )
        gpu = cp.asnumpy(cp.asarray(handle))[0]

        assert_close(gpu, cpu, rtol=2e-1, atol=2e-1, msg="CUDA ADXR batch mismatch")

    def test_adxr_cuda_many_series_one_param_matches_cpu(self, test_data):
        T = 1500
        N = 3
        base_h = test_data["high"][:T].astype(np.float64)
        base_l = test_data["low"][:T].astype(np.float64)
        base_c = test_data["close"][:T].astype(np.float64)
        high_tm = np.zeros((T, N), dtype=np.float64)
        low_tm = np.zeros((T, N), dtype=np.float64)
        close_tm = np.zeros((T, N), dtype=np.float64)
        for j in range(N):
            high_tm[:, j] = base_h * (1.0 + 0.01 * j)
            low_tm[:, j] = base_l * (1.0 + 0.01 * j)
            close_tm[:, j] = base_c * (1.0 + 0.01 * j)

        period = 14

        cpu_tm = np.zeros_like(close_tm)
        for j in range(N):
            cpu_tm[:, j] = ti.adxr(high_tm[:, j], low_tm[:, j], close_tm[:, j], period)

        handle = ti.adxr_cuda_many_series_one_param_dev(
            high_tm.astype(np.float32),
            low_tm.astype(np.float32),
            close_tm.astype(np.float32),
            period,
        )
        gpu_tm = cp.asnumpy(cp.asarray(handle))

        assert gpu_tm.shape == cpu_tm.shape
        assert_close(gpu_tm, cpu_tm, rtol=2e-1, atol=2e-1, msg="CUDA ADXR ms1p mismatch")

