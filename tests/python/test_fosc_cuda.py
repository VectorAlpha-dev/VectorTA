"""Python binding tests for FOSC CUDA kernels.
Skips gracefully when CUDA is unavailable or CUDA feature not built.
"""
import numpy as np
import pytest

try:
    import cupy as cp
except ImportError:  # pragma: no cover
    cp = None

try:
    import my_project as ti
except ImportError:  # pragma: no cover
    pytest.skip(
        "Python module not built. Run 'maturin develop --features python,cuda' first",
        allow_module_level=True,
    )

from test_utils import assert_close


def _cuda_available() -> bool:
    if cp is None:
        return False
    if not hasattr(ti, "fosc_cuda_batch_dev"):
        return False
    try:
        x = np.array([np.nan, 1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        handle = ti.fosc_cuda_batch_dev(x, period_range=(3, 3, 0))
        _ = cp.asarray(handle)  # ensure CuPy can wrap the handle
        return True
    except Exception as exc:  # pragma: no cover
        msg = str(exc).lower()
        if "cuda not available" in msg or "ptx" in msg:
            return False
        return True


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available or cuda bindings not built")
class TestFoscCuda:
    def test_fosc_cuda_batch_matches_cpu(self):
        # Create a synthetic series with NaNs at the front
        n = 4096
        price = np.full(n, np.nan, dtype=np.float64)
        for i in range(5, n):
            x = float(i)
            price[i] = np.sin(x * 0.00123) + 0.00011 * x

        sweep = (8, 32, 4)
        cpu = ti.fosc_batch(price, period_range=sweep)
        cpu_vals = cpu["values"].astype(np.float32)

        handle = ti.fosc_cuda_batch_dev(price.astype(np.float32), period_range=sweep)
        gpu_vals = cp.asnumpy(cp.asarray(handle)).reshape(cpu_vals.shape)

        # Slightly relaxed tolerance (fp32 vs fp64)
        assert_close(gpu_vals, cpu_vals, rtol=8e-4, atol=8e-4, msg="FOSC CUDA batch mismatch")

    def test_fosc_cuda_many_series_one_param_matches_cpu(self):
        T, N = 2048, 6
        data_tm = np.full((T, N), np.nan, dtype=np.float64)
        for s in range(N):
            start = s % 5
            for t in range(start, T):
                x = float(t) + 0.37 * s
                data_tm[t, s] = np.sin(x * 0.0019) + 0.00021 * x

        period = 14
        # CPU baseline per series
        cpu_tm = np.full_like(data_tm, np.nan)
        for s in range(N):
            cpu_tm[:, s] = ti.fosc(data_tm[:, s], period)

        handle = ti.fosc_cuda_many_series_one_param_dev(
            data_tm.astype(np.float32), period
        )
        gpu_tm = cp.asnumpy(cp.asarray(handle))

        assert gpu_tm.shape == data_tm.shape
        assert_close(gpu_tm, cpu_tm.astype(np.float32), rtol=8e-4, atol=8e-4, msg="FOSC CUDA many-series mismatch")

