"""Python CUDA binding tests for CVI (Chaikin's Volatility)."""
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
    if not hasattr(ti, "cvi_cuda_batch_dev"):
        return False
    try:
        data = load_test_data()
        h = data["high"][:128].astype(np.float32)
        l = data["low"][:128].astype(np.float32)
        handle = ti.cvi_cuda_batch_dev(h, l, (10, 10, 0))
        _ = cp.asarray(handle)
        return True
    except Exception as exc:  
        msg = str(exc).lower()
        if "cuda not available" in msg or "nvcc" in msg or "ptx" in msg:
            return False
        return True


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available or bindings not built")
class TestCviCuda:
    @pytest.fixture(scope="class")
    def dataset(self):
        return load_test_data()

    def test_batch_matches_cpu(self, dataset):
        high = dataset["high"].astype(np.float64)
        low = dataset["low"].astype(np.float64)

        periods = [5, 10, 20]
        handle = ti.cvi_cuda_batch_dev(
            high.astype(np.float32),
            low.astype(np.float32),
            (periods[0], periods[-1], 5),
        )
        gpu = cp.asnumpy(cp.asarray(handle))
        assert gpu.shape == (len(periods), high.shape[0])

        
        for row, p in enumerate(periods):
            cpu = ti.cvi(high, low, p)
            assert_close(gpu[row], cpu, rtol=1e-6, atol=2e-3,
                         msg=f"CVI CUDA batch mismatch (p={p})")

    def test_many_series_one_param_matches_cpu(self, dataset):
        T = 1024
        N = 4
        
        close = dataset["close"][:T].astype(np.float64)
        data_tm = np.zeros((T, N), dtype=np.float64)
        for j in range(N):
            data_tm[:, j] = close * (1.0 + 0.01 * j)
        
        x = np.arange(T, dtype=np.float64)
        off = (0.004 * (0.002 * x)).reshape(T, 1)
        high_tm = data_tm + (0.10 + np.abs(off))
        low_tm  = data_tm - (0.10 + np.abs(off))

        period = 14

        
        cpu_tm = np.zeros_like(data_tm)
        for j in range(N):
            cpu_tm[:, j] = ti.cvi(high_tm[:, j], low_tm[:, j], period)

        
        handle = ti.cvi_cuda_many_series_one_param_dev(
            high_tm.astype(np.float32).ravel(),
            low_tm.astype(np.float32).ravel(),
            N,
            T,
            period,
        )
        gpu_tm = cp.asnumpy(cp.asarray(handle))
        assert gpu_tm.shape == cpu_tm.shape
        assert_close(gpu_tm, cpu_tm, rtol=1e-6, atol=2e-3,
                     msg="CVI CUDA many-series vs CPU mismatch")

