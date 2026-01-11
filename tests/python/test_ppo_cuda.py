"""CUDA bindings tests for PPO (Percentage Price Oscillator)."""
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
    if not hasattr(ti, "ppo_cuda_batch_dev"):
        return False
    try:
        arr = np.array([np.nan, 1.0, 2.0, 3.0], dtype=np.float32)
        handle, _meta = ti.ppo_cuda_batch_dev(arr, (5, 5, 0), (10, 10, 0), "sma")
        _ = cp.asarray(handle)
        return True
    except Exception as exc:  
        msg = str(exc).lower()
        if "cuda not available" in msg or "ptx" in msg or "driver" in msg:
            return False
        return True


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available or cuda bindings missing")
class TestPpoCuda:
    @pytest.fixture(scope="class")
    def series(self):
        data = load_test_data()["close"].astype(np.float64)
        return data

    def test_ppo_cuda_batch_matches_cpu_sma(self, series):
        fast, slow = 12, 26
        cpu = ti.ppo(series, fast, slow, "sma")
        handle, meta = ti.ppo_cuda_batch_dev(series.astype(np.float32), (fast, fast, 0), (slow, slow, 0), "sma")
        assert meta["fast_periods"].tolist() == [fast]
        assert meta["slow_periods"].tolist() == [slow]
        gpu = cp.asnumpy(cp.asarray(handle))[0]
        assert gpu.shape[0] == series.shape[0]
        assert_close(gpu, cpu, rtol=2e-3, atol=2e-3, msg="CUDA PPO SMA batch mismatch")

    def test_ppo_cuda_many_series_one_param_matches_cpu_ema(self, series):
        
        T = 2048
        N = 4
        base = series[:T]
        data_tm = np.zeros((T, N), dtype=np.float64)
        for j in range(N):
            data_tm[:, j] = base * (1.0 + 0.01 * j)
        fast, slow = 10, 21
        
        cpu_tm = np.zeros_like(data_tm)
        for j in range(N):
            cpu_tm[:, j] = ti.ppo(data_tm[:, j], fast, slow, "ema")
        handle = ti.ppo_cuda_many_series_one_param_dev(
            data_tm.astype(np.float32), fast, slow, "ema"
        )
        gpu_tm = cp.asnumpy(cp.asarray(handle))
        assert gpu_tm.shape == data_tm.shape
        assert_close(gpu_tm, cpu_tm, rtol=3e-3, atol=3e-3, msg="CUDA PPO EMA many-series mismatch")

