"""
Python binding tests for Donchian CUDA kernels.
Skips automatically when CUDA is unavailable or bindings are not built.
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
    if not hasattr(ti, "donchian_cuda_batch_dev"):
        return False
    try:
        
        high = np.array([np.nan, 3.0, 5.0, 4.0], dtype=np.float32)
        low  = np.array([np.nan, 1.0, 2.0, 3.0], dtype=np.float32)
        out = ti.donchian_cuda_batch_dev(high, low, period_range=(2, 2, 0))
        _ = cp.asarray(out["upper"])  
        return True
    except Exception as exc:  
        msg = str(exc).lower()
        if "cuda not available" in msg or "ptx" in msg or "nvcc" in msg:
            return False
        return True


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available or cuda bindings not built")
class TestDonchianCuda:
    @pytest.fixture(scope="class")
    def test_data(self):
        return load_test_data()

    def test_donchian_cuda_batch_matches_cpu(self, test_data):
        high = test_data["high"].astype(np.float64)
        low  = test_data["low"].astype(np.float64)
        period = 20

        cpu = ti.donchian(high, low, period)
        cpu_upper = cpu["upperband"]
        cpu_middle = cpu["middleband"]
        cpu_lower = cpu["lowerband"]

        out = ti.donchian_cuda_batch_dev(
            high.astype(np.float32),
            low.astype(np.float32),
            period_range=(period, period, 0),
        )
        g_upper = cp.asnumpy(cp.asarray(out["upper"]))[0]
        g_middle = cp.asnumpy(cp.asarray(out["middle"]))[0]
        g_lower = cp.asnumpy(cp.asarray(out["lower"]))[0]

        assert_close(g_upper, cpu_upper, rtol=5e-4, atol=5e-4, msg="upper mismatch")
        assert_close(g_middle, cpu_middle, rtol=5e-4, atol=5e-4, msg="middle mismatch")
        assert_close(g_lower, cpu_lower, rtol=5e-4, atol=5e-4, msg="lower mismatch")

    def test_donchian_cuda_many_series_one_param_matches_cpu(self, test_data):
        rows = 1024
        cols = 4
        high_base = test_data["high"][:rows].astype(np.float64)
        low_base  = test_data["low"][:rows].astype(np.float64)
        high_tm = np.zeros((rows, cols), dtype=np.float64)
        low_tm  = np.zeros((rows, cols), dtype=np.float64)
        for j in range(cols):
            high_tm[:, j] = high_base + 0.05 * j
            low_tm[:, j]  = low_base  - 0.03 * j
            high_tm[: j + 8, j] = np.nan
            low_tm[: j + 8, j] = np.nan

        period = 18
        cpu_u = np.zeros_like(high_tm)
        cpu_m = np.zeros_like(high_tm)
        cpu_l = np.zeros_like(high_tm)
        for j in range(cols):
            out = ti.donchian(high_tm[:, j], low_tm[:, j], period)
            cpu_u[:, j] = out["upperband"]
            cpu_m[:, j] = out["middleband"]
            cpu_l[:, j] = out["lowerband"]

        out = ti.donchian_cuda_many_series_one_param_dev(
            high_tm.astype(np.float32),
            low_tm.astype(np.float32),
            period,
        )
        g_u = cp.asnumpy(cp.asarray(out["upper"]))
        g_m = cp.asnumpy(cp.asarray(out["middle"]))
        g_l = cp.asnumpy(cp.asarray(out["lower"]))

        assert g_u.shape == (rows, cols)
        assert g_m.shape == (rows, cols)
        assert g_l.shape == (rows, cols)

        mask = ~np.isnan(cpu_u)
        assert_close(g_u[mask], cpu_u[mask], rtol=5e-4, atol=5e-4, msg="upper many-series mismatch")
        mask = ~np.isnan(cpu_m)
        assert_close(g_m[mask], cpu_m[mask], rtol=5e-4, atol=5e-4, msg="middle many-series mismatch")
        mask = ~np.isnan(cpu_l)
        assert_close(g_l[mask], cpu_l[mask], rtol=5e-4, atol=5e-4, msg="lower many-series mismatch")

