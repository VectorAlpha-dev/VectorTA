"""Python binding tests for UltOsc CUDA kernels."""
import numpy as np
import pytest

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
    if not hasattr(ti, "ultosc_cuda_batch_dev"):
        return False
    try:
        
        h = np.array([np.nan, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        l = np.array([np.nan, 1.0, 2.5, 3.5, 4.0], dtype=np.float32)
        c = np.array([np.nan, 1.5, 2.8, 3.8, 4.5], dtype=np.float32)
        handle = ti.ultosc_cuda_batch_dev(
            h, l, c,
            (7, 7, 0), (14, 14, 0), (28, 28, 0)
        )
        _ = cp.asarray(handle)
        return True
    except Exception as exc:  
        msg = str(exc).lower()
        if "cuda not available" in msg or "ptx" in msg or "nvcc" in msg:
            return False
        return True


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available or cuda bindings not built")
class TestUltoscCuda:
    def test_ultosc_cuda_batch_matches_cpu(self):
        n = 4096
        x = np.arange(n, dtype=np.float64)
        base = np.sin(x * 0.00123) + 0.00017 * x
        spread = np.abs(np.cos(x * 0.0027)) * 0.05 + 0.02
        h = base + spread
        l = base - spread
        c = base.copy()
        h[:2] = np.nan; l[:2] = np.nan; c[:2] = np.nan

        sweep1 = (5, 25, 10)
        sweep2 = (10, 30, 10)
        sweep3 = (20, 40, 10)

        cpu = ti.ultosc_batch(h, l, c, sweep1, sweep2, sweep3)
        cpu_vals = cpu["values"].astype(np.float32)

        handle = ti.ultosc_cuda_batch_dev(h.astype(np.float32),
                                          l.astype(np.float32),
                                          c.astype(np.float32),
                                          sweep1, sweep2, sweep3)
        gpu_vals = cp.asnumpy(cp.asarray(handle)).reshape(cpu_vals.shape)

        mask = ~np.isnan(cpu_vals)
        assert_close(gpu_vals[mask], cpu_vals[mask], rtol=3e-4, atol=5e-4,
                     msg="CUDA UltOsc mismatch vs CPU baseline")

    def test_ultosc_cuda_many_series_one_param_matches_cpu(self):
        if not hasattr(ti, "ultosc_cuda_many_series_one_param_dev"):
            pytest.skip("ultosc_cuda_many_series_one_param_dev not available")

        cols = 6; rows = 2048
        h = np.full((rows, cols), np.nan, dtype=np.float64)
        l = np.full((rows, cols), np.nan, dtype=np.float64)
        c = np.full((rows, cols), np.nan, dtype=np.float64)
        for s in range(cols):
            for t in range(1, rows):
                xx = float(t) + float(s) * 0.41
                base = np.sin(xx * 0.002) + 0.0003 * xx
                sp = np.abs(np.cos(xx * 0.0013)) * 0.04 + 0.01
                h[t, s] = base + sp
                l[t, s] = base - sp
                c[t, s] = base

        p1, p2, p3 = 7, 14, 28

        
        cpu = np.full((rows, cols), np.nan, dtype=np.float64)
        for s in range(cols):
            series_h, series_l, series_c = h[:, s], l[:, s], c[:, s]
            cpu[:, s] = ti.ultosc(series_h, series_l, series_c, p1, p2, p3)

        handle = ti.ultosc_cuda_many_series_one_param_dev(h.astype(np.float32),
                                                          l.astype(np.float32),
                                                          c.astype(np.float32),
                                                          cols, rows, p1, p2, p3)
        gpu = cp.asnumpy(cp.asarray(handle)).astype(np.float64)

        mask = ~np.isnan(cpu)
        assert_close(gpu[mask], cpu[mask], rtol=3e-4, atol=5e-4,
                     msg="CUDA UltOsc many-series mismatch vs CPU baseline")

