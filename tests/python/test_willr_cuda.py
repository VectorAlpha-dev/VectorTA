"""Python CUDA binding tests for Williams' %R (WILLR)."""
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
    if not hasattr(ti, "willr_cuda_batch_dev"):
        return False
    try:
        data = load_test_data()
        high = data["high"][:128].astype(np.float32)
        low = data["low"][:128].astype(np.float32)
        close = data["close"][:128].astype(np.float32)
        handle = ti.willr_cuda_batch_dev(high, low, close, period_range=(14, 14, 0))
        _ = cp.asarray(handle)
        return True
    except Exception as exc:  
        msg = str(exc).lower()
        if "cuda not available" in msg or "nvcc" in msg or "ptx" in msg:
            return False
        return True


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available or cuda bindings not built")
class TestWillrCuda:
    @pytest.fixture(scope="class")
    def dataset(self):
        return load_test_data()

    def test_willr_cuda_batch_matches_cpu(self, dataset):
        high = dataset["high"].astype(np.float64)
        low = dataset["low"].astype(np.float64)
        close = dataset["close"].astype(np.float64)

        periods = [14, 21, 28]
        handle = ti.willr_cuda_batch_dev(
            high.astype(np.float32),
            low.astype(np.float32),
            close.astype(np.float32),
            period_range=(periods[0], periods[-1], periods[1] - periods[0]),
        )

        gpu = cp.asnumpy(cp.asarray(handle))
        assert gpu.shape[0] == len(periods)
        assert gpu.shape[1] == high.shape[0]

        for row, period in enumerate(periods):
            cpu = ti.willr(high, low, close, period)
            assert_close(
                gpu[row],
                cpu,
                rtol=1e-5,
                atol=2e-5,
                msg=f"CUDA batch vs CPU mismatch (period={period})",
            )

    def test_willr_cuda_many_series_one_param_matches_cpu(self, dataset):
        T = 1024
        N = 6
        period = 14

        base = dataset["close"][:T].astype(np.float64)
        high_tm = np.full((T, N), np.nan, dtype=np.float64)
        low_tm = np.full((T, N), np.nan, dtype=np.float64)
        close_tm = np.full((T, N), np.nan, dtype=np.float64)
        for j in range(N):
            fv = min(j, 5)
            off = 0.1 + 0.02 * j
            for t in range(fv, T):
                v = base[t] if np.isfinite(base[t]) else (0.0)
                high_tm[t, j] = v + off
                low_tm[t, j] = v - off
                close_tm[t, j] = v

        cpu_tm = np.full_like(close_tm, np.nan)
        for j in range(N):
            cpu_tm[:, j] = ti.willr(high_tm[:, j], low_tm[:, j], close_tm[:, j], period)

        handle = ti.willr_cuda_many_series_one_param_dev(
            high_tm.astype(np.float32).ravel(),
            low_tm.astype(np.float32).ravel(),
            close_tm.astype(np.float32).ravel(),
            N,
            T,
            period,
        )
        gpu_tm = cp.asnumpy(cp.asarray(handle))

        assert gpu_tm.shape == cpu_tm.shape
        assert_close(
            gpu_tm,
            cpu_tm,
            rtol=2e-5,
            atol=5e-5,
            msg="WILLR CUDA many-series mismatch",
        )
