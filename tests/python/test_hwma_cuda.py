"""Python binding tests for HWMA CUDA kernels."""
import pytest
import numpy as np

try:
    import cupy as cp
except ImportError:
    cp = None

try:
    import vector_ta as ti
except ImportError:
    pytest.skip(
        "Python module not built. Run 'maturin develop --features python,cuda' first",
        allow_module_level=True,
    )

from test_utils import assert_close, load_test_data


def _cuda_available() -> bool:
    if cp is None:
        return False
    if not hasattr(ti, "hwma_cuda_batch_dev"):
        return False
    try:
        probe = np.array([np.nan, np.nan, 1.0, 2.0, 3.0], dtype=np.float64)
        handle = ti.hwma_cuda_batch_dev(probe, (0.1, 0.1, 0.0), (0.1, 0.1, 0.0), (0.1, 0.1, 0.0))
        _ = cp.asarray(handle)
        return True
    except Exception as exc:
        msg = str(exc).lower()
        if "cuda not available" in msg or "ptx" in msg or "nvcc" in msg:
            return False
        return True


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available or cuda bindings not built")
class TestHwmaCuda:
    @pytest.fixture(scope="class")
    def price_series(self):
        data = load_test_data()
        limit = 4096
        arr = data["close"][:limit].astype(np.float64).copy()
        arr[:20] = np.nan
        return arr

    def test_hwma_cuda_batch_matches_cpu(self, price_series):
        sweep = ((0.1, 0.4, 0.1), (0.05, 0.25, 0.05), (0.05, 0.2, 0.05))

        cpu = ti.hwma_batch(price_series, *sweep)
        cpu_values = np.asarray(cpu["values"], dtype=np.float64)

        handle = ti.hwma_cuda_batch_dev(price_series, *sweep)
        gpu = cp.asnumpy(cp.asarray(handle))

        assert gpu.shape == cpu_values.shape
        stable = np.isnan(gpu) | (np.abs(gpu) < 1e5)
        assert np.count_nonzero(~stable) / gpu.size < 0.1
        close = np.isclose(gpu[stable], cpu_values[stable], rtol=6e-4, atol=8e-4, equal_nan=True)
        assert np.count_nonzero(~close) / close.size < 0.001

    def test_hwma_cuda_many_series_one_param_matches_cpu(self, price_series):
        T = 1024
        N = 4
        series = price_series[:T]
        data_tm = np.zeros((T, N), dtype=np.float64)
        for j in range(N):
            data_tm[:, j] = series * (1.0 + 0.03 * j)

        na, nb, nc = 0.18, 0.12, 0.08

        cpu_tm = np.zeros_like(data_tm)
        for j in range(N):
            cpu_tm[:, j] = ti.hwma(np.ascontiguousarray(data_tm[:, j]), na, nb, nc)

        handle = ti.hwma_cuda_many_series_one_param_dev(
            data_tm.astype(np.float32), na, nb, nc
        )
        gpu_tm = cp.asnumpy(cp.asarray(handle))

        assert gpu_tm.shape == data_tm.shape
        assert_close(gpu_tm, cpu_tm, rtol=6e-4, atol=8e-4, msg="CUDA many-series vs CPU mismatch")
