"""Python binding tests for Wilders CUDA kernels."""
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

from test_utils import load_test_data, assert_close


def _cuda_available() -> bool:
    if cp is None:
        return False
    if not hasattr(ti, "wilders_cuda_batch_dev"):
        return False
    try:
        sample = np.array([np.nan, 1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        handle = ti.wilders_cuda_batch_dev(sample, period_range=(3, 3, 0))
        _ = cp.asarray(handle)
        return True
    except Exception as exc:
        msg = str(exc).lower()
        if "cuda not available" in msg or "nvcc" in msg or "ptx" in msg:
            return False
        return True


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available or cuda bindings not built")
class TestWildersCuda:
    @pytest.fixture(scope="class")
    def test_data(self):
        return load_test_data()

    def test_wilders_cuda_batch_matches_cpu(self, test_data):
        close = test_data["close"]
        period = 14

        cpu = ti.wilders(close, period)

        handle = ti.wilders_cuda_batch_dev(
            close.astype(np.float32),
            period_range=(period, period, 0),
        )

        gpu = cp.asarray(handle)
        gpu_first = cp.asnumpy(gpu)[0]

        assert_close(gpu_first, cpu, rtol=1e-5, atol=1e-6, msg="Wilders CUDA batch vs CPU mismatch")

    def test_wilders_cuda_batch_sweep_matches_cpu(self, test_data):
        close = test_data["close"][:2048]
        sweep = (5, 25, 2)

        cpu_rows = []
        for period in range(sweep[0], sweep[1] + 1, sweep[2]):
            cpu_rows.append(ti.wilders(close, period))
        cpu = np.vstack(cpu_rows)

        handle = ti.wilders_cuda_batch_dev(
            close.astype(np.float32),
            period_range=sweep,
        )
        gpu = cp.asnumpy(cp.asarray(handle))

        assert gpu.shape == cpu.shape
        assert_close(gpu, cpu, rtol=1e-5, atol=1e-6, msg="Wilders CUDA sweep mismatch")
