"""Python binding tests for WAD CUDA kernels."""
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
    if not hasattr(ti, "wad_cuda_batch_dev"):
        return False
    try:
        close = np.linspace(100.0, 101.0, 16, dtype=np.float32)
        high = close + 0.2
        low = close - 0.2
        handle = ti.wad_cuda_batch_dev(high, low, close)  
        _ = cp.asarray(handle)  
        return True
    except Exception as exc:  
        msg = str(exc).lower()
        if "cuda not available" in msg or "ptx" in msg:
            return False
        return True


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available or cuda bindings not built")
class TestWadCuda:
    def _make_series(self, n: int = 2048):
        t = np.arange(n, dtype=np.float64)
        base = 100.0 + np.sin(t * 0.0041) * 0.7 + np.cos(t * 0.0023) * 0.4
        close = base
        high = close + 0.65 + 0.05 * (t % 5)
        low = close - 0.64 - 0.04 * (t % 3)
        return high, low, close

    def test_wad_cuda_batch_matches_cpu(self):
        high, low, close = self._make_series()
        cpu = ti.wad(high, low, close)
        cpu_vals = cpu.astype(np.float32)
        handle = ti.wad_cuda_batch_dev(high.astype(np.float32), low.astype(np.float32), close.astype(np.float32))
        gpu_vals = cp.asnumpy(cp.asarray(handle)).reshape(cpu_vals.shape)
        assert_close(gpu_vals, cpu_vals, rtol=3e-4, atol=3e-4, msg="CUDA WAD mismatch")

