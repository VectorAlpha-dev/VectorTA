"""Python binding tests for Linear Regression Angle CUDA kernels."""
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

from test_utils import assert_close


def _cuda_available() -> bool:
    if cp is None:
        return False
    if not hasattr(ti, "linearreg_angle_cuda_batch_dev"):
        return False

    try:
        x = np.linspace(0.0, 1.0, 128, dtype=np.float32)
        handle = ti.linearreg_angle_cuda_batch_dev(x, period_range=(8, 8, 0))
        _ = cp.asarray(handle)
        return True
    except Exception as e:
        msg = str(e).lower()
        if "cuda not available" in msg or "ptx" in msg or "nvcc" in msg:
            return False
        return True


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available or cuda bindings not built")
class TestLinearregAngleCuda:
    def test_lra_cuda_batch_matches_cpu(self):
        n = 4096
        x = np.arange(n, dtype=np.float64)
        series = np.sin(0.0023 * x) + 0.00017 * x
        period = 20

        cpu = ti.linearreg_angle(series, period)
        handle = ti.linearreg_angle_cuda_batch_dev(series.astype(np.float32), (period, period, 0))
        gpu = cp.asnumpy(cp.asarray(handle))[0].astype(np.float64)


        warm = period - 1
        mask = ~np.isnan(cpu)
        assert_close(gpu[mask], cpu[mask], rtol=1e-4, atol=5e-4,
                     msg="LRA CUDA batch mismatch vs CPU baseline")

    def test_lra_cuda_many_series_one_param_matches_cpu(self):
        rows = 2048
        cols = 6
        x = np.arange(rows, dtype=np.float64)
        tm = np.stack([np.sin(0.003 * x) + 0.0004 * x * (1.0 + 0.05 * j) for j in range(cols)], axis=1)
        period = 21

        cpu_tm = np.stack([ti.linearreg_angle(tm[:, j], period) for j in range(cols)], axis=1)
        handle = ti.linearreg_angle_cuda_many_series_one_param_dev(tm.astype(np.float32), cols, rows, period)
        gpu_tm = cp.asnumpy(cp.asarray(handle)).astype(np.float64)


        mask = ~np.isnan(cpu_tm)
        assert_close(gpu_tm[mask], cpu_tm[mask], rtol=1e-4, atol=5e-4,
                     msg="LRA CUDA many-series mismatch vs CPU baseline")

