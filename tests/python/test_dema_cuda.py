"""Python binding tests for DEMA CUDA kernels."""
import pytest
import numpy as np

try:
    import cupy as cp
except ImportError:  # pragma: no cover
    cp = None

try:
    import my_project as ti
except ImportError:
    pytest.skip("Python module not built. Run 'maturin develop --features python,cuda' first", allow_module_level=True)

from test_utils import load_test_data, assert_close


def _cuda_available() -> bool:
    if cp is None:
        return False
    if not hasattr(ti, "dema_cuda_batch_dev"):
        return False
    try:
        sample = np.array([np.nan, 1.0, 2.0, 3.0], dtype=np.float32)
        handle = ti.dema_cuda_batch_dev(sample, period_range=(3, 3, 0))
        _ = cp.asarray(handle)
        return True
    except Exception as exc:  # pragma: no cover - best effort probe
        msg = str(exc).lower()
        if "cuda not available" in msg or "nvcc" in msg or "ptx" in msg:
            return False
        return True


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available or cuda bindings not built")
class TestDemaCuda:
    @pytest.fixture(scope="class")
    def test_data(self):
        return load_test_data()

    def test_dema_cuda_batch_matches_cpu(self, test_data):
        close = test_data["close"]
        period = 30

        cpu = ti.dema(close, period)

        handle = ti.dema_cuda_batch_dev(
            close.astype(np.float32),
            period_range=(period, period, 0),
        )

        gpu = cp.asarray(handle)
        gpu_first = cp.asnumpy(gpu)[0]

        # Tighter tolerances thanks to FMA delta updates in CUDA kernel
        assert_close(gpu_first, cpu, rtol=5e-6, atol=5e-7, msg="DEMA CUDA batch vs CPU mismatch")

    def test_dema_cuda_batch_multiple_periods(self, test_data):
        close = test_data["close"][:2048]
        sweep = (10, 20, 2)
        cpu_rows = []
        for period in range(sweep[0], sweep[1] + 1, sweep[2]):
            cpu_rows.append(ti.dema(close, period))
        cpu = np.vstack(cpu_rows)

        handle = ti.dema_cuda_batch_dev(
            close.astype(np.float32),
            period_range=sweep,
        )

        gpu = cp.asnumpy(cp.asarray(handle))
        assert gpu.shape == cpu.shape
        assert_close(gpu, cpu, rtol=5e-6, atol=5e-7, msg="DEMA CUDA sweep mismatch")
