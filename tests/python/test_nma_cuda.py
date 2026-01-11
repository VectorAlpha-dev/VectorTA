"""CUDA bindings tests for the Normalized Moving Average (NMA)."""
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
    if not hasattr(ti, "nma_cuda_batch_dev"):
        return False
    try:
        arr = np.array([np.nan, 1.5, 2.0, 2.5], dtype=np.float32)
        handle, _meta = ti.nma_cuda_batch_dev(arr, period_range=(3, 3, 0))
        _ = cp.asarray(handle)
        return True
    except Exception as exc:  
        msg = str(exc).lower()
        if "cuda not available" in msg or "ptx" in msg or "driver" in msg:
            return False
        return True


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available or cuda bindings missing")
class TestNmaCuda:
    @pytest.fixture(scope="class")
    def test_data(self):
        return load_test_data()

    def test_nma_cuda_batch_matches_cpu(self, test_data):
        close = test_data["close"].astype(np.float64)
        pos_close = np.abs(close) + 1.0
        period = 16

        cpu = ti.nma(pos_close, period)

        handle, meta = ti.nma_cuda_batch_dev(
            pos_close.astype(np.float32),
            period_range=(period, period, 0),
        )
        assert meta["periods"].tolist() == [period]

        gpu = cp.asnumpy(cp.asarray(handle))[0]
        assert_close(gpu, cpu, rtol=1e-5, atol=1e-6, msg="CUDA NMA batch mismatch")

    def test_nma_cuda_many_series_one_param_matches_cpu(self, test_data):
        base = test_data["close"][:1024].astype(np.float64)
        pos_base = np.abs(base) + 1.0
        n_series = 4
        data_tm = np.zeros((pos_base.shape[0], n_series), dtype=np.float64)
        for j in range(n_series):
            data_tm[:, j] = pos_base * (1.0 + 0.015 * j)

        period = 18

        cpu_tm = np.zeros_like(data_tm)
        for j in range(n_series):
            cpu_tm[:, j] = ti.nma(data_tm[:, j], period)

        handle = ti.nma_cuda_many_series_one_param_dev(
            data_tm.astype(np.float32), period
        )
        gpu_tm = cp.asnumpy(cp.asarray(handle))

        assert gpu_tm.shape == data_tm.shape
        assert_close(
            gpu_tm,
            cpu_tm,
            rtol=1e-5,
            atol=1e-6,
            msg="CUDA NMA many-series mismatch",
        )
