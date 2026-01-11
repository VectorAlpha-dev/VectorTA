"""
Python binding tests for Deviation CUDA kernels (stddev path).
Skips gracefully when CUDA or CuPy is unavailable, or when bindings are absent.
"""
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
    if not hasattr(ti, "deviation_cuda_batch_dev"):
        return False
    
    try:
        data = np.array([np.nan, 1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        handle, meta = ti.deviation_cuda_batch_dev(data, period_range=(3, 3, 0))
        _ = cp.asarray(handle)
        return True
    except Exception as exc:  
        msg = str(exc).lower()
        if "cuda not available" in msg or "ptx" in msg or "nvcc" in msg:
            return False
        return True


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available or cuda bindings not built")
class TestDeviationCuda:
    def test_deviation_cuda_batch_matches_cpu(self):
        n = 4096
        data = np.full(n, np.nan, dtype=np.float64)
        for i in range(6, n):
            x = float(i)
            base = np.sin(x * 0.00041) - np.cos(x * 0.00017)
            data[i] = base + 0.0013 * ((i % 9) - 4)
            if i % 257 == 0:
                data[i] = np.nan

        sweep = (10, 40, 10)
        cpu = ti.deviation_batch(data, sweep, (0, 0, 0))
        gpu_handle, meta = ti.deviation_cuda_batch_dev(data.astype(np.float32), period_range=sweep)
        gpu = cp.asnumpy(cp.asarray(gpu_handle))

        assert gpu.shape == (len(meta["periods"]), data.shape[0])
        assert gpu.shape == (cpu["values"].shape[0], cpu["values"].shape[1])

        assert_close(
            gpu, cpu["values"], rtol=5e-4, atol=5e-4, msg="Deviation CUDA vs CPU mismatch"
        )

