"""
Python binding tests for KAUFMANSTOP CUDA kernels.
Skips gracefully when CUDA is unavailable or CUDA bindings not built.
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
    pytest.skip("Python module not built. Run 'maturin develop --features python,cuda' first", allow_module_level=True)

from test_utils import assert_close


def _cuda_available() -> bool:
    if cp is None:
        return False
    if not hasattr(ti, 'kaufmanstop_cuda_batch_dev'):
        return False
    
    try:
        high = np.array([np.nan, 2.0, 3.0, 4.0], dtype=np.float32)
        low = np.array([np.nan, 1.0, 1.5, 2.5], dtype=np.float32)
        handle = ti.kaufmanstop_cuda_batch_dev(
            high, low, period_range=(3, 3, 0), mult_range=(2.0, 2.0, 0.0), direction="long", ma_type="sma"
        )
        _ = cp.asarray(handle)
        return True
    except Exception as e:
        msg = str(e).lower()
        if 'cuda not available' in msg or 'nvcc' in msg or 'ptx' in msg:
            return False
        return True


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available or cuda bindings not built")
class TestKaufmanstopCuda:
    def test_kaufmanstop_cuda_batch_matches_cpu(self):
        n = 4096
        x = np.arange(n, dtype=np.float64)
        base = np.sin(x * 0.002) + 0.001 * x
        rng = 0.5 + np.abs(np.cos(x * 0.0007))
        high = base + 0.5 * rng
        low = base - 0.5 * rng

        period = 22
        mult = 2.0
        direction = 'long'
        ma_type = 'sma'

        
        cpu = ti.kaufmanstop(high, low, period=period, mult=mult, direction=direction, ma_type=ma_type)

        
        handle = ti.kaufmanstop_cuda_batch_dev(
            high.astype(np.float32),
            low.astype(np.float32),
            period_range=(period, period, 0),
            mult_range=(mult, mult, 0.0),
            direction=direction,
            ma_type=ma_type,
        )
        gpu_row = cp.asnumpy(cp.asarray(handle))[0]

        assert_close(gpu_row, cpu, rtol=5e-5, atol=1e-5, msg="CUDA batch vs CPU mismatch")

    def test_kaufmanstop_cuda_many_series_one_param_matches_cpu(self):
        rows = 1024
        cols = 4
        x = np.arange(rows, dtype=np.float64)
        base = np.sin(x * 0.002) + 0.001 * x
        rng = 0.5 + np.abs(np.cos(x * 0.0007))
        hi = base + 0.5 * rng
        lo = base - 0.5 * rng
        high_tm = np.stack([hi * (1.0 + 0.01 * j) for j in range(cols)], axis=1)
        low_tm = np.stack([lo * (1.0 + 0.01 * j) for j in range(cols)], axis=1)

        period = 22
        mult = 2.0
        direction = 'long'
        ma_type = 'sma'

        cpu_tm = np.zeros_like(high_tm)
        for j in range(cols):
            cpu_tm[:, j] = ti.kaufmanstop(high_tm[:, j], low_tm[:, j], period=period, mult=mult, direction=direction, ma_type=ma_type)

        handle = ti.kaufmanstop_cuda_many_series_one_param_dev(
            high_tm.astype(np.float32),
            low_tm.astype(np.float32),
            period,
            mult,
            direction,
            ma_type,
        )
        gpu_tm = cp.asnumpy(cp.asarray(handle))
        assert gpu_tm.shape == cpu_tm.shape
        assert_close(gpu_tm, cpu_tm, rtol=5e-5, atol=1e-5, msg="CUDA many-series vs CPU mismatch")

