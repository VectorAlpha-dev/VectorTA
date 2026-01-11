"""CUDA bindings tests for the Simple Moving Average (SMA)."""
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
    if not hasattr(ti, "sma_cuda_batch_dev"):
        return False
    try:
        arr = np.array([np.nan, 1.0, 2.0, 3.0], dtype=np.float32)
        handle, _meta = ti.sma_cuda_batch_dev(arr, period_range=(3, 3, 0))
        _ = cp.asarray(handle)
        return True
    except Exception as exc:  
        msg = str(exc).lower()
        if "cuda not available" in msg or "ptx" in msg or "driver" in msg:
            return False
        return True


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available or cuda bindings missing")
class TestSmaCuda:
    @pytest.fixture(scope="class")
    def test_data(self):
        return load_test_data()

    def test_sma_cuda_batch_matches_cpu(self, test_data):
        close = test_data["close"].astype(np.float64)
        period = 14

        cpu = ti.sma(close, period)

        handle, meta = ti.sma_cuda_batch_dev(
            close.astype(np.float32),
            period_range=(period, period, 0),
        )
        assert meta["periods"].tolist() == [period]

        gpu = cp.asnumpy(cp.asarray(handle))[0]
        assert_close(gpu, cpu, rtol=1e-5, atol=1e-6, msg="CUDA SMA batch mismatch")

    def test_sma_cuda_many_series_one_param_matches_cpu(self, test_data):
        T = 1024
        N = 4
        base = test_data["close"][:T].astype(np.float64)
        data_tm = np.zeros((T, N), dtype=np.float64)
        for j in range(N):
            data_tm[:, j] = base * (1.0 + 0.02 * j)

        period = 10

        cpu_tm = np.zeros_like(data_tm)
        for j in range(N):
            cpu_tm[:, j] = ti.sma(data_tm[:, j], period)

        handle = ti.sma_cuda_many_series_one_param_dev(
            data_tm.astype(np.float32), period
        )
        gpu_tm = cp.asnumpy(cp.asarray(handle))

        assert gpu_tm.shape == data_tm.shape
        assert_close(
            gpu_tm,
            cpu_tm,
            rtol=1e-5,
            atol=1e-6,
            msg="CUDA SMA many-series mismatch",
        )
