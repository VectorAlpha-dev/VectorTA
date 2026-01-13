"""
Python binding tests for EDCF CUDA kernels.
Mirrors the CPU batch API and validates GPU results against the scalar baseline.
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
    pytest.skip("Python module not built. Run 'maturin develop --features python,cuda' first", allow_module_level=True)

from test_utils import assert_close, load_test_data


def _cuda_available() -> bool:
    if cp is None:
        return False
    if not hasattr(ti, "edcf_cuda_batch_dev"):
        return False
    try:
        x = np.array([np.nan, 1.0, 2.0, 3.0], dtype=np.float32)
        handle = ti.edcf_cuda_batch_dev(x, period_range=(5, 5, 0))
        _ = cp.asarray(handle)
        return True
    except Exception as exc:
        msg = str(exc).lower()
        if "cuda not available" in msg or "nvcc" in msg or "ptx" in msg:
            return False
        return True


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available or cuda bindings not built")
class TestEdcfCuda:
    @pytest.fixture(scope="class")
    def test_data(self):
        return load_test_data()

    def test_edcf_cuda_batch_matches_cpu(self, test_data):
        close = test_data["close"]
        period_range = (15, 39, 4)

        cpu = ti.edcf_batch(close, period_range)
        cpu_vals = cpu["values"]

        handle = ti.edcf_cuda_batch_dev(close.astype(np.float32), period_range)
        gpu = cp.asnumpy(cp.asarray(handle))

        assert gpu.shape == cpu_vals.shape
        assert np.array_equal(np.isnan(gpu), np.isnan(cpu_vals))

        mask = ~np.isnan(cpu_vals)
        assert_close(gpu[mask], cpu_vals[mask], rtol=1e-4, atol=1e-5, msg="CUDA batch vs CPU mismatch")
