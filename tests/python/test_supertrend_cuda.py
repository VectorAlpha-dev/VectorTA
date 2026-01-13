"""
Python CUDA binding tests for SuperTrend.
Skips gracefully when CUDA or bindings are unavailable.
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
    if not hasattr(ti, "supertrend_cuda_batch_dev"):
        return False
    try:
        data = load_test_data()
        high = data["high"][:128].astype(np.float32)
        low = data["low"][:128].astype(np.float32)
        close = data["close"][:128].astype(np.float32)
        out = ti.supertrend_cuda_batch_dev(high, low, close, (10, 10, 0), (3.0, 3.0, 0.0))
        _ = cp.asarray(out["trend"])
        _ = cp.asarray(out["changed"])
        return True
    except Exception as exc:
        msg = str(exc).lower()
        if "cuda not available" in msg or "ptx" in msg or "device" in msg:
            return False
        return True


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available or cuda bindings not built")
class TestSupertrendCuda:
    @pytest.fixture(scope="class")
    def ds(self):
        return load_test_data()

    def test_supertrend_cuda_batch_matches_cpu(self, ds):
        high = ds["high"].astype(np.float64)
        low = ds["low"].astype(np.float64)
        close = ds["close"].astype(np.float64)

        period = 10
        factor = 3.0

        trend_cpu, changed_cpu = ti.supertrend(high, low, close, period, factor)

        gpu = ti.supertrend_cuda_batch_dev(
            high.astype(np.float32),
            low.astype(np.float32),
            close.astype(np.float32),
            (period, period, 0),
            (factor, factor, 0.0),
        )

        trend_gpu = cp.asnumpy(cp.asarray(gpu["trend"]))[0]
        changed_gpu = cp.asnumpy(cp.asarray(gpu["changed"]))[0]

        assert_close(trend_gpu, trend_cpu, rtol=1e-4, atol=2e-3, msg="trend mismatch")
        assert_close(changed_gpu, changed_cpu, rtol=1e-6, atol=1e-7, msg="changed mismatch")

    def test_supertrend_cuda_many_series_one_param_matches_cpu(self, ds):
        rows = 2048
        cols = 4
        close = ds["close"][:rows].astype(np.float64)
        series = np.vstack([close * (1.0 + 0.01 * j) for j in range(cols)]).T


        high = series + 0.12 + 0.004 * np.sin(np.arange(rows) * 0.002)[:, None]
        low = series - 0.12 - 0.004 * np.sin(np.arange(rows) * 0.002)[:, None]

        period = 10
        factor = 3.0

        trend_cpu = np.zeros_like(series)
        changed_cpu = np.zeros_like(series)
        for j in range(cols):
            t, ch = ti.supertrend(high[:, j], low[:, j], series[:, j], period, factor)
            trend_cpu[:, j] = t
            changed_cpu[:, j] = ch

        out = ti.supertrend_cuda_many_series_one_param_dev(
            high.astype(np.float32),
            low.astype(np.float32),
            series.astype(np.float32),
            cols,
            rows,
            period,
            factor,
        )

        trend_tm = cp.asnumpy(cp.asarray(out["trend"]))
        changed_tm = cp.asnumpy(cp.asarray(out["changed"]))

        assert_close(trend_tm, trend_cpu, rtol=1e-4, atol=2e-3, msg="trend TM mismatch")
        assert_close(changed_tm, changed_cpu, rtol=1e-6, atol=1e-7, msg="changed TM mismatch")

