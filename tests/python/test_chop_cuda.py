"""Python binding tests for CHOP CUDA kernels."""
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
    if not hasattr(ti, "chop_cuda_batch_dev"):
        return False
    try:

        n = 64
        h = np.linspace(1, 2, n, dtype=np.float32)
        l = np.linspace(0.5, 1.5, n, dtype=np.float32)
        c = (h + l) * 0.5
        handle = ti.chop_cuda_batch_dev(h, l, c, (5, 5, 0), (100.0, 100.0, 0.0), (1, 1, 0))
        _ = cp.asarray(handle)
        return True
    except Exception as exc:
        msg = str(exc).lower()
        if "cuda not available" in msg or "ptx" in msg or "nvcc" in msg:
            return False
        return True


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available or cuda bindings not built")
class TestChopCuda:
    def test_chop_cuda_batch_matches_cpu(self):
        data = load_test_data()
        h = data["high"].astype(np.float64)
        l = data["low"].astype(np.float64)
        c = data["close"].astype(np.float64)

        sweep = (5, 25, 5)
        drifts = (1, 3, 1)

        cpu = ti.chop_batch(h, l, c, sweep, (100.0, 100.0, 0.0), drifts)
        cpu_values = cpu["values"]

        handle = ti.chop_cuda_batch_dev(h.astype(np.float32), l.astype(np.float32), c.astype(np.float32), sweep, (100.0, 100.0, 0.0), drifts)
        gpu = cp.asnumpy(cp.asarray(handle))

        assert gpu.shape == cpu_values.shape
        mask = ~np.isnan(cpu_values)
        assert_close(gpu[mask], cpu_values[mask], rtol=1e-4, atol=5e-3,
                     msg="CUDA CHOP mismatch vs CPU baseline")

